import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF Generator", layout="wide")

# =====================================================
# CONSTANTS (Excel / VBA Gumbel)
# =====================================================
EULER_GAMMA = 0.5772156649
SIGMA_Y = 1.28255  # Std dev of reduced variate (Excel)

# =====================================================
# Helper: sort files by numeric suffix (_1, _2, ...)
# =====================================================
def sort_files_by_numeric_suffix(files):
    def extract_index(f):
        m = re.search(r'_(\d+)', f.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =====================================================
# Read rainfall data (UPLOAD ONLY)
# =====================================================
@st.cache_data(show_spinner="Reading uploaded rainfall data...")
def read_rainfall_from_upload(files):
    frames = []

    for f in files:
        df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
        df.columns = df.columns.str.lower()

        time_col = next(c for c in df.columns if "time" in c or "date" in c)
        rain_col = next(c for c in df.columns if "rain" in c)

        df = df[[time_col, rain_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce")

        df.dropna(inplace=True)
        df.columns = ["time", "rain"]

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # ✅ chronological ordering
    merged.sort_values("time", inplace=True)

    return merged["time"].to_numpy(), merged["rain"].to_numpy()

# =====================================================
# AMS (VBA compatible)
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    window = int(duration_min / interval_min)
    years = pd.DatetimeIndex(times).year.to_numpy()

    cumsum = np.zeros(len(rain) + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, len(rain)):
        value = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])

        if yr not in ams:
            ams[yr] = value
        elif value > ams[yr]:
            ams[yr] = value

    return np.array(list(ams.values()), dtype=float)

# =====================================================
# Distributions
# =====================================================
def gumbel_excel_q(x, T):
    xbar = x.mean()           # AVERAGE
    s = x.std(ddof=1)         # STDEV.S
    yT = -np.log(np.log(T / (T - 1.0)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y
    return xbar + KT * s

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

# =====================================================
# UI
# =====================================================
st.title("🌧️ AMS / DDF / IDF Generator")

with st.sidebar:
    interval = st.number_input(
        "Data interval (minutes)",
        min_value=1,
        value=6
    )

    durations = st.multiselect(
        "Durations (minutes)",
        [5, 10, 15, 30, 60, 120, 360, 720, 1440],
        default=[30, 60, 120, 360, 720, 1440]
    )

    if "return_periods" not in st.session_state:
        st.session_state.return_periods = [2, 5, 10, 20, 30, 50, 100]

    selected_T = st.multiselect(
        "Return periods (years)",
        st.session_state.return_periods,
        default=[2, 5, 10, 20, 30, 50, 100]
    )

    custom_T = st.text_input("Add custom return periods (years)")
    if custom_T:
        for t in custom_T.split(","):
            t = t.strip()
            if t.isdigit():
                ti = int(t)
                if ti not in st.session_state.return_periods:
                    st.session_state.return_periods.append(ti)
        st.session_state.return_periods = sorted(st.session_state.return_periods)

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    files = st.file_uploader(
        "Upload rainfall files (CSV / Excel)",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    btn_ams = st.button("✅ Compute AMS")
    btn_ddf = st.button("📐 Compute DDF & IDF")

# =====================================================
# Require uploaded data
# =====================================================
if not files:
    st.info("Please upload rainfall data files to begin.")
    st.stop()

files = sort_files_by_numeric_suffix(files)
times, rain = read_rainfall_from_upload(files)

# =====================================================
# AMS
# =====================================================
if btn_ams:
    ams_data = {}
    for d in durations:
        ams_data[d] = compute_ams_vba(times, rain, d, interval)

    st.session_state["AMS_DATA"] = ams_data

    ams_df = pd.DataFrame(
        {f"AMS_{d}min": ams_data[d] for d in durations}
    ).round(2)

    st.dataframe(ams_df, use_container_width=True)

# =====================================================
# DDF / IDF
# =====================================================
if btn_ddf:

    if "AMS_DATA" not in st.session_state:
        st.warning("Please compute AMS first.")
        st.stop()

    for dist in distributions:
        st.subheader(f"📐 DDF & IDF – {dist}")

        ddf = {}
        for d, x in st.session_state["AMS_DATA"].items():
            values = []
            for T in selected_T:
                if dist == "Gumbel":
                    values.append(gumbel_excel_q(x, T))
                elif dist == "GEV":
                    values.append(gev_q(x, T))
                elif dist == "LP-III":
                    values.append(lp3_q(x, T))
                elif dist == "Lognormal":
                    values.append(lognormal_q(x, T))
            ddf[d] = values

        ddf_df = pd.DataFrame(ddf, index=selected_T).T.round(2)
        ddf_df.index.name = "Duration (min)"

        st.markdown("**Rainfall Depth (mm)**")
        st.dataframe(ddf_df, use_container_width=True)

        idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0).round(2)

        st.markdown("**Rainfall Intensity (mm/hr)**")
        st.dataframe(idf_df, use_container_width=True)
