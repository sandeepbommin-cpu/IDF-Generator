import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="AMS Generator (VBA-Compatible)",
    layout="wide"
)

# =====================================================
# AUTO-SORT FILES BY NUMERIC SUFFIX (_1, _2, _3, ...)
# =====================================================
def sort_files_by_numeric_suffix(files):
    """
    Sort uploaded files by numeric suffix in filename.
    Example: Rainfall_Data_1, Rainfall_Data_2, ...
    Files without a suffix go last.
    """
    def extract_index(f):
        match = re.search(r'_(\d+)', f.name)
        return int(match.group(1)) if match else float("inf")

    return sorted(files, key=extract_index)

# =====================================================
# FAST, CACHED, VBA-STYLE RAINFALL READER
# =====================================================
@st.cache_data(show_spinner="Reading rainfall data...")
def read_rainfall_vba_style_cached(files):
    """
    Fast rainfall reader:
    - Preserves file + row order
    - Vectorized datetime & numeric parsing
    - Skips invalid rows (VBA IsDate behaviour)
    """
    all_times = []
    all_rain = []

    for f in files:
        if f.name.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        df.columns = df.columns.str.lower()

        time_col = next(c for c in df.columns if "time" in c or "date" in c)
        rain_col = next(c for c in df.columns if "rain" in c)

        times = pd.to_datetime(df[time_col], errors="coerce")
        rain = pd.to_numeric(df[rain_col], errors="coerce")

        mask = times.notna() & rain.notna()
        all_times.append(times[mask])
        all_rain.append(rain[mask])

    times = pd.concat(all_times, ignore_index=True).to_numpy()
    rain = pd.concat(all_rain, ignore_index=True).to_numpy()

    return times, rain

# =====================================================
# OPTIMIZED VBA-COMPATIBLE AMS (O(N))
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    """
    EXACT VBA semantics:
    - Rolling window over row order
    - Year assigned at END of window
    - O(N) using cumulative sums
    """
    window = int(duration_min / interval_min)
    n = len(rain)

    # Convert times once to obtain years safely
    years = pd.DatetimeIndex(times).year.to_numpy()

    # Prefix sum for fast rolling sums
    cumsum = np.zeros(n + 1, dtype=float)
    cumsum[1:] = np.cumsum(rain)

    ams = {}

    for i in range(window - 1, n):
        window_sum = cumsum[i + 1] - cumsum[i + 1 - window]
        year = int(years[i])  # END-of-window year (VBA behaviour)

        if year not in ams:
            ams[year] = window_sum
        else:
            if window_sum > ams[year]:
                ams[year] = window_sum

    return ams

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🌧️ Annual Maximum Series (AMS)")
st.caption("Fast, VBA-compatible AMS generator with auto-sorted inputs")

with st.sidebar:
    st.header("Inputs")

    interval = st.number_input(
        "Data interval (minutes)",
        min_value=1,
        value=6
    )

    durations = st.multiselect(
        "Durations (minutes)",
        [5, 10, 15, 30, 60, 120, 360, 720, 1440],
        default=[60, 1440]
    )

    files = st.file_uploader(
        "Upload rainfall data files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("🔄 Clear rainfall data"):
        st.session_state.pop("rain_data", None)
        st.cache_data.clear()
        st.experimental_rerun()

# =====================================================
# LOAD / REUSE RAINFALL DATA
# =====================================================
if "rain_data" not in st.session_state:
    st.session_state["rain_data"] = None

if files and st.session_state["rain_data"] is None:
    # Auto-sort files by numeric suffix
    files = sort_files_by_numeric_suffix(files)

    # Display detected order (for transparency)
    with st.sidebar:
        st.markdown("**Detected file order:**")
        for f in files:
            st.write(f.name)

    times, rain = read_rainfall_vba_style_cached(files)
    st.session_state["rain_data"] = (times, rain)
    st.success(f"Rainfall data loaded and cached: {len(rain):,} records")

if st.session_state["rain_data"] is None:
    st.info("Upload rainfall files to compute AMS.")
    st.stop()

times, rain = st.session_state["rain_data"]

if len(rain) == 0:
    st.error("No valid rainfall records found.")
    st.stop()

# =====================================================
# COMPUTE & DISPLAY AMS
# =====================================================
st.subheader("📊 Annual Maximum Series (AMS)")

ams_table = {}

for d in durations:
    ams_dict = compute_ams_vba(times, rain, d, interval)
    ams_table[f"{d} min"] = pd.Series(ams_dict)

ams_df = pd.DataFrame(ams_table)
ams_df.index.name = "Year"
ams_df.sort_index(inplace=True)

st.dataframe(ams_df, use_container_width=True)

st.download_button(
    "Download AMS CSV",
    ams_df.to_csv().encode(),
    "AMS.csv"
)
