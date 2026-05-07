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
# Helper: sort uploaded files by numeric suffix
# =====================================================
def sort_files_by_numeric_suffix(files):
    def extract_index(f):
        m = re.search(r'_(\d+)', f.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =====================================================
# Read rainfall data (UPLOAD ONLY, VBA-compatible)
# =====================================================
@st.cache_data(show_spinner="Reading uploaded rainfall data...")
def read_rainfall_from_upload(files):
    all_times, all_rain = [], []

    for f in files:
        df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
        df.columns = df.columns.str.lower()

        tcol = next(c for c in df.columns if "time" in c or "date" in c)
        rcol = next(c for c in df.columns if "rain" in c)

        t = pd.to_datetime(df[tcol], errors="coerce")
        r = pd.to_numeric(df[rcol], errors="coerce")

        mask = t.notna() & r.notna()
        all_times.append(t[mask])
        all_rain.append(r[mask])

    return (
        pd.concat(all_times, ignore_index=True).to_numpy(),
        pd.concat(all_rain, ignore_index=True).to_numpy()
    )

# =====================================================
# AMS (EXACT VBA LOGIC)
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
        elif wsum > ams[yr]:
            ams[yr] = wsum

    return np.array(list(ams.values()), dtype=float)

# =====================================================
# Excel/VBA Gumbel
# =====================================================
def gumbel_excel_q(x, T):
    xbar = x.mean()
    s = x.std(ddof=1)
    yT = -np.log(np.log(T / (T - 1.0)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y
    return xbar + KT * s

# =====================================================
# Alternating Block Method (ABM)
# =====================================================
def alternating_block_method(total_depth, duration_min, timestep_min):
    n = int(duration_min / timestep_min)
    cumulative = np.linspace(timestep_min, duration_min, n) / duration_min * total_depth
    increments = np.diff(np.insert(cumulative, 0, 0))

    order = np.argsort(increments)[::-1]
    blocks = increments[order]

    hyeto = np.zeros(n)
    center = n // 2
    hyeto[center] = blocks[0]

    idx = 1
    for i in range(1, n):
        pos = center + idx if i % 2 == 1 else center - idx
        if pos < 0 or pos >= n:
            idx += 1
            pos = center + idx if i % 2 == 1 else center - idx
        hyeto[pos] = blocks[i]
        if i % 2 == 0:
            idx += 1

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
    T = st.selectbox("Return Period for ABM (years)", return_periods)

    files = st.file_uploader(
        "Upload rainfall files (CSV / Excel)",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    run = st.button("✅ Run AMS → DDF → ABM")

# =====================================================
# Execution
# =====================================================
if run:

    if not files:
        st.warning("Upload rainfall files first.")
        st.stop()

    files = sort_files_by_numeric_suffix(files)
    times, rain = read_rainfall_from_upload(files)

    for d in durations:
        ams = compute_ams_vba(times, rain, d, interval)
        depth = gumbel_excel_q(ams, T)

        hyeto = alternating_block_method(depth, d, interval)
        time = np.arange(interval, d + interval, interval)

        df = pd.DataFrame({
            "Time (min)": time,
            "Incremental Rainfall (mm)": hyeto,
            "Cumulative Rainfall (mm)": np.cumsum(hyeto).round(2)
        })

        st.subheader(f"ABM Hyetograph — {d} min, T = {T} years")
        st.dataframe(df, use_container_width=True)
