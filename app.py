import streamlit as st
import pandas as pd
import numpy as np
import re
import pathlib
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF Generator", layout="wide")

DATA_DIR = pathlib.Path("data")

# =========================
# Constants
# =========================
EULER_GAMMA = 0.5772156649015329
GUMBEL_SD_FACTOR = np.sqrt(6) / np.pi
FROZEN_DURATIONS = [30, 60, 120, 360, 720, 1440]
DATA_INTERVAL_MIN = 6

# =========================
# File sorting
# =========================
def sort_files_by_numeric_suffix(files):
    def extract_index(p):
        m = re.search(r'_(\d+)', p.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =========================
# Read rainfall data
# =========================
@st.cache_data(show_spinner="Reading rainfall data from repository...")
def read_rainfall_from_repo():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise RuntimeError("No rainfall files found in data/ directory")

    files = sort_files_by_numeric_suffix(files)

    all_times = []
    all_rain = []

    for f in files:
        df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
        df.columns = df.columns.str.lower()

        time_col = next(c for c in df.columns if "time" in c or "date" in c)
        rain_col = next(c for c in df.columns if "rain" in c)

        t = pd.to_datetime(df[time_col], errors="coerce")
        r = pd.to_numeric(df[rain_col], errors="coerce")

        mask = t.notna() & r.notna()
        all_times.append(t[mask])
        all_rain.append(r[mask])

    times = pd.concat(all_times, ignore_index=True).to_numpy()
    rain = pd.concat(all_rain, ignore_index=True).to_numpy()

    return times, rain, files

# =========================
# AMS (VBA-compatible)
# =========================
def compute_ams_vba(times, rain, duration_min):
    window = int(duration_min / DATA_INTERVAL_MIN)
    n = len(rain)

    years = pd.DatetimeIndex(times).year.to_numpy()
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, n):
        wsum = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])

        if yr not in ams:
            ams[yr] = wsum
        elif wsum > ams[yr]:
            ams[yr] = wsum

    return np.array(list(ams.values()))

# =========================
# Distributions
# =========================
def gumbel_mom_q(x, T):
    mu = x.mean()
    sigma = x.std(ddof=0)
    beta = sigma / GUMBEL_SD_FACTOR
    alpha = mu - EULER_GAMMA * beta
    yT = -np.log(-np.log(1.0 - 1.0 / T))
    return alpha + beta * yT

def gev_q(x, T):
    lm = distr.gev.lmom_fit(x)
    return distr.gev.ppf(1 - 1 / T, **lm)

def lp3_q(x, T):
    lx = np.log(x)
    params = pearson3.fit(lx)
    return np.exp(pearson3.ppf(1 - 1 / T, *params))

def lognormal_q(x, T):
    shape, loc, scale = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1 / T, shape, loc, scale)

# =========================
# Load data & freeze AMS
# =========================
times, rain, used_files = read_rainfall_from_repo()

if "AMS_DATA" not in st.session_state:
    st.session_state["AMS_DATA"] = {}
    for d in FROZEN_DURATIONS:
        st.session_state["AMS_DATA"][d] = compute_ams_vba(times, rain, d)

# =========================
# UI
# =========================
st.title("🌧️ AMS / DDF / IDF Generator")
st.caption("AMS frozen for testing; DDF/IDF derived from AMS")

with st.sidebar:
    st.header("Return periods")

    if "return_periods" not in st.session_state:
        st.session_state["return_periods"] = [2, 5, 10, 20, 50, 100]

    selected_freqs = st.multiselect(
        "Return periods (years)",
        st.session_state["return_periods"],
        default=[2, 10, 50, 100]
    )

    custom_freq_text = st.text_input(
        "Add custom return periods",
        placeholder="e.g. 25, 30"
    )

    if custom_freq_text:
        for f in custom_freq_text.split(","):
            f = f.strip()
            if f.isdigit() and int(f) not in st.session_state["return_periods"]:
                st.session_state["return_periods"].append(int(f))
        st.session_state["return_periods"] = sorted(st.session_state["return_periods"])

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    run_button = st.button("📐 Compute DDF & IDF")

st.sidebar.markdown("**Rainfall files used:**")
for f in used_files:
    st.sidebar.write(f.name)

# =========================
# DDF / IDF
# =========================
if run_button and distributions and selected_freqs:

    for dist in distributions:
        st.subheader(f"📐 DDF & IDF – {dist}")

        ddf = {}
        for d in FROZEN_DURATIONS:
            x = st.session_state["AMS_DATA"][d]
            values = []

            for T in selected_freqs:
                if dist == "Gumbel":
                    values.append(gumbel_mom_q(x, T))
                elif dist == "GEV":
                    values.append(gev_q(x, T))
                elif dist == "LP-III":
                    values.append(lp3_q(x, T))
                elif dist == "Lognormal":
                    values.append(lognormal_q(x, T))

            ddf[d] = values

        ddf_df = pd.DataFrame(ddf, index=selected_freqs).T
        st.markdown("**Rainfall Depth (mm)**")
        st.dataframe(ddf_df, use_container_width=True)

        idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0)
        st.markdown("**Rainfall Intensity (mm/hr)**")
        st.dataframe(idf_df, use_container_width=True)
