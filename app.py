import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AMS Generator (VBA-Compatible)",
    layout="wide"
)

# =====================================================
# CACHED VBA-STYLE DATA READER
# =====================================================
@st.cache_data(show_spinner="Reading rainfall data...")
def read_rainfall_vba_style_cached(files):
    """
    FAST, VBA-compatible rainfall reader.
    - Preserves row order
    - Skips invalid timestamps
    - Uses vectorized Pandas ops (no Python loops)
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

        # ✅ Vectorized parsing
        times = pd.to_datetime(df[time_col], errors="coerce")
        rain = pd.to_numeric(df[rain_col], errors="coerce")

        mask = times.notna() & rain.notna()
        times = times[mask]
        rain = rain[mask]

        all_times.append(times)
        all_rain.append(rain)

    # ✅ Concatenate once (preserves file + row order)
    times = pd.concat(all_times, ignore_index=True).to_numpy()
    rain = pd.concat(all_rain, ignore_index=True).to_numpy()

    return times, rain


# =====================================================
# OPTIMIZED VBA-STYLE AMS COMPUTATION (O(N))
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    """
    Optimized AMS computation:
    - Rolling window using cumulative sum
    - Row order preserved
    - Year assigned at END of window
    """
    window = int(duration_min / interval_min)
    n = len(rain)

    # Prefix sum for fast rolling sums
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}

    for i in range(window - 1, n):
        window_sum = cumsum[i + 1] - cumsum[i + 1 - window]
        year = times[i].year

        if year not in ams:
            ams[year] = window_sum
        elif window_sum > ams[year]:
            ams[year] = window_sum

    return ams


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🌧️ Annual Maximum Series (AMS)")
st.caption("Optimized, VBA-compatible AMS computation")

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
        "Upload rainfall data files (once)",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("🔄 Clear rainfall data"):
        if "rain_data" in st.session_state:
            del st.session_state["rain_data"]
        st.cache_data.clear()
        st.experimental_rerun()


# =====================================================
# LOAD / RETRIEVE RAINFALL DATA
# =====================================================
if "rain_data" not in st.session_state:
    st.session_state["rain_data"] = None

if files and st.session_state["rain_data"] is None:
    times, rain = read_rainfall_vba_style_cached(files)
    st.session_state["rain_data"] = (times, rain)
    st.success("Rainfall data loaded and cached.")

if st.session_state["rain_data"] is None:
    st.info("Upload rainfall files to generate AMS.")
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
