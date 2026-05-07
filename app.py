import streamlit as st
import pandas as pd
import numpy as np
import re
import pathlib
from scipy.stats import gumbel_r, lognorm, pearson3
from lmoments3 import distr

st.set_page_config(
    page_title="AMS / DDF / IDF Generator",
    layout="wide"
)

DATA_DIR = pathlib.Path("data")

# =====================================================
# FILE ORDERING (_1, _2, _3 ...)
# =====================================================
def sort_files_by_numeric_suffix(file_paths):
    def extract_index(p):
        m = re.search(r'_(\d+)', p.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(file_paths, key=extract_index)

# =====================================================
# READ RAINFALL FROM REPOSITORY
# =====================================================
@st.cache_data(show_spinner="Reading rainfall data from repository...")
def read_rainfall_from_repo():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No rainfall files found in data/ folder")

    files = sort_files_by_numeric_suffix(files)

    all_times, all_rain = [], []

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
    rain  = pd.concat(all_rain, ignore_index=True).to_numpy()

    return times, rain, files

# =====================================================
# AMS (VBA-COMPATIBLE, OPTIMIZED)
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    window = int(duration_min / interval_min)
    n = len(rain)

    years = pd.DatetimeIndex(times).year.to_numpy()

    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, n):
        window_sum = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])  # END-of-window year (VBA behaviour)

        if yr not in ams:
            ams[yr] = window_sum
        elif window_sum > ams[yr]:
            ams[yr] = window_sum

    return ams

# =====================================================
# DISTRIBUTION QUANTILES
# =====================================================
def gumbel_q(x, T):
    loc, scale = gumbel_r.fit(x)
    return gumbel_r.ppf(1 - 1/T, loc, scale)

def gev_q(x, T):
    lm = distr.gev.lmom_fit(x)
    return distr.gev.ppf(1 - 1/T, **lm)

def lp3_q(x, T):
    lx = np.log(x)
    params = pearson3.fit(lx)
    return np.exp(pearson3.ppf(1 - 1/T, *params))

def lognormal_q(x, T):
    shape, loc, scale = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1/T, shape, loc, scale)

# =====================================================
# UI
# =====================================================
st.title("🌧️ AMS / DDF / IDF Generator")

with st.sidebar:
    st.header("Inputs")

    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)

    base_durations = [5,10,15,30,60,120,360,720,1440]
    selected_durations = st.multiselect("Durations (minutes)", base_durations)

    custom_duration_text = st.text_input(
        "Add custom duration (minutes)",
        placeholder="e.g. 180"
    )

    freq_defaults = [2, 5, 10, 20, 50, 100]
    selected_freqs = st.multiselect("Return periods (years)", freq_defaults)

    custom_freq_text = st.text_input(
        "Add custom return periods (years)",
        placeholder="e.g. 25, 30, 75"
    )

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    compute_ams_btn = st.button("✅ Compute AMS")
    compute_ddf_btn = st.button("📐 Compute DDF & IDF")

# =====================================================
# LOAD DATA
# =====================================================
times, rain, used_files = read_rainfall_from_repo()

st.sidebar.markdown("**Rainfall files used:**")
for f in used_files:
    st.sidebar.write(f.name)

# =====================================================
# AMS
# =====================================================
if compute_ams_btn:

    durations = selected_durations.copy()
    if custom_duration_text.strip():
        durations.append(int(custom_duration_text))

    if not durations:
        st.warning("Select at least one duration.")
        st.stop()

    st.subheader("📊 Annual Maximum Series (AMS)")

    ams_table = {}
    for d in sorted(set(durations)):
        ams = compute_ams_vba(times, rain, d, interval)
        ams_table[f"{d} min"] = pd.Series(ams)

    ams_df = pd.DataFrame(ams_table)
    ams_df.index.name = "Year"
    ams_df.sort_index(inplace=True)

    st.dataframe(ams_df, use_container_width=True)
    st.download_button("Download AMS CSV", ams_df.to_csv().encode(), "AMS.csv")

# =====================================================
# DDF / IDF
# =====================================================
if compute_ddf_btn:

    durations = selected_durations.copy()
    if custom_duration_text.strip():
        durations.append(int(custom_duration_text))

    freqs = selected_freqs.copy()
    if custom_freq_text.strip():
        freqs.extend(int(f) for f in custom_freq_text.split(","))

    durations = sorted(set(durations))
    freqs = sorted(set(freqs))

    if not durations or not freqs or not distributions:
        st.warning("Select durations, frequencies, and distributions.")
        st.stop()

    for dist in distributions:
        st.subheader(f"📐 DDF & IDF – {dist}")

        ddf = {}
        for d in durations:
            ams_vals = list(compute_ams_vba(times, rain, d, interval).values())
            x = np.array(ams_vals)

            vals = []
            for T in freqs:
                if dist == "Gumbel":
                    vals.append(gumbel_q(x, T))
                elif dist == "GEV":
                    vals.append(gev_q(x, T))
                elif dist == "LP-III":
                    vals.append(lp3_q(x, T))
                elif dist == "Lognormal":
                    vals.append(lognormal_q(x, T))

            ddf[d] = vals

        ddf_df = pd.DataFrame(ddf, index=freqs).T
        ddf_df.index.name = "Duration (min)"
        ddf_df.columns = [f"T={t}" for t in freqs]

        st.markdown("**Rainfall Depth (mm)**")
        st.dataframe(ddf_df, use_container_width=True)

        idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0)

        st.markdown("**Rainfall Intensity (mm/hr)**")
        st.dataframe(idf_df, use_container_width=True)
``
