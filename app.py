import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF / ABM Generator", layout="wide")

# =====================================================
# CONSTANTS (Excel / VBA Gumbel)
# =====================================================
EULER_GAMMA = 0.5772156649
SIGMA_Y = 1.28255

# =====================================================
# Helper: sort uploaded files by numeric suffix (_1, _2, ...)
# =====================================================
def sort_files_by_numeric_suffix(files):
    def extract_index(f):
        m = re.search(r'_(\d+)', f.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =====================================================
# Read rainfall data (UPLOAD ONLY, VBA-compatible order)
# =====================================================
@st.cache_data(show_spinner="Reading uploaded rainfall data...")
def read_rainfall_from_upload(files):
    times, rain = [], []

    for f in files:
        df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
        df.columns = df.columns.str.lower()

        tcol = next(c for c in df.columns if "time" in c or "date" in c)
        rcol = next(c for c in df.columns if "rain" in c)

        t = pd.to_datetime(df[tcol], errors="coerce")
        r = pd.to_numeric(df[rcol], errors="coerce")

        mask = t.notna() & r.notna()
        times.append(t[mask])
        rain.append(r[mask])

    return (
        pd.concat(times, ignore_index=True).to_numpy(),
        pd.concat(rain, ignore_index=True).to_numpy()
    )

# =====================================================
# AMS (EXACT VBA LOGIC – syntax fixed)
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    window = int(duration_min / interval_min)
    years = pd.DatetimeIndex(times).year.to_numpy()

    cumsum = np.zeros(len(rain) + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, len(rain)):
        wsum = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])

        if yr not in ams:
            ams[yr] = wsum
        else:
            if wsum > ams[yr]:
                ams[yr] = wsum

    return np.array(list(ams.values()), dtype=float)

# =====================================================
# Distributions
# =====================================================
def gumbel_excel_q(x, T):
    mean = x.mean()
    s = x.std(ddof=1)  # STDEV.S
    yT = -np.log(np.log(T / (T - 1.0)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y
    return mean + KT * s

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
# Alternating Block Method (ABM)
# =====================================================
def alternating_block_method(total_depth, duration_min, timestep_min):
    n = int(duration_min / timestep_min)

    cumulative = np.linspace(timestep_min, duration_min, n) / duration_min * total_depth
    inc = np.diff(np.insert(cumulative, 0, 0))

    blocks = np.sort(inc)[::-1]

    hyeto = np.zeros(n)
    center = n // 2
    hyeto[center] = blocks[0]

    left, right = center - 1, center + 1
    for i in range(1, len(blocks)):
        if i % 2 == 1 and right < n:
            hyeto[right] = blocks[i]
            right += 1
        elif left >= 0:
            hyeto[left] = blocks[i]
            left -= 1

    return hyeto.round(2)

# =====================================================
# UI
# =====================================================
st.title("🌧️ AMS / DDF / IDF / ABM Generator")

with st.sidebar:
    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)

    durations = st.multiselect(
        "Durations (minutes)",
        [30, 60, 120, 360, 720, 1440],
        default=[60]
    )

    return_periods = [2, 5, 10, 20, 30, 50, 100]
    selected_T = st.multiselect(
        "Return periods",
        return_periods,
        default=return_periods
    )

    files = st.file_uploader(
        "Upload rainfall files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    st.divider()

    abm_durations = st.multiselect(
        "ABM durations (minutes)",
        [30, 60, 120, 360, 720, 1440],
        default=[60]
    )

    abm_T = st.multiselect(
        "ABM return periods",
        return_periods,
        default=[10]
    )

    btn_ams = st.button("Compute AMS")
    btn_ddf = st.button("Compute DDF / IDF")
    btn_abm = st.button("Compute ABM")

# =====================================================
# Require uploaded data
# =====================================================
if not files:
    st.info("Upload rainfall data to begin.")
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
    st.subheader("AMS")
    st.dataframe(pd.DataFrame(ams_data).round(2))

# =====================================================
# DDF / IDF
# =====================================================
if btn_ddf:
    if "AMS_DATA" not in st.session_state:
        st.warning("Compute AMS first.")
        st.stop()

    ddf = {}
    for d, x in st.session_state["AMS_DATA"].items():
        ddf[d] = [gumbel_excel_q(x, T) for T in selected_T]

    ddf_df = pd.DataFrame(ddf, index=selected_T).T.round(2)
    st.session_state["DDF_DATA"] = ddf_df

    st.subheader("DDF (mm)")
    st.dataframe(ddf_df)

    st.subheader("IDF (mm/hr)")
    st.dataframe(ddf_df.div(ddf_df.index / 60.0, axis=0).round(2))

# =====================================================
# ABM
# =====================================================
if btn_abm:
    if "DDF_DATA" not in st.session_state:
        st.warning("Compute DDF / IDF first.")
        st.stop()

    for d in abm_durations:
        for T in abm_T:
            depth = st.session_state["DDF_DATA"].loc[d, T]
            h = alternating_block_method(depth, d, interval)

            df = pd.DataFrame({
                "Time (min)": np.arange(interval, d + interval, interval),
                "Incremental Rainfall (mm)": h,
                "Cumulative Rainfall (mm)": np.cumsum(h).round(2)
            })

            st.subheader(f"ABM Hyetograph – {d} min, T={T} years")
            st.dataframe(df)
