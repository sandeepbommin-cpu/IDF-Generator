import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AMS Generator (VBA-Compatible)",
    layout="wide"
)

# =====================================================
# VBA-STYLE DATA READER
# =====================================================
def read_rainfall_vba_style(files):
    """
    Reads rainfall files in upload order.
    Does NOT sort by time.
    Skips non-date rows (VBA IsDate behaviour).
    """
    times = []
    rainfall = []

    for f in files:
        if f.name.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        df.columns = df.columns.str.lower()

        time_col = [c for c in df.columns if "time" in c or "date" in c][0]
        rain_col = [c for c in df.columns if "rain" in c][0]

        for _, row in df.iterrows():
            try:
                t = pd.to_datetime(row[time_col])
                r = float(row[rain_col])
                times.append(t)
                rainfall.append(r)
            except Exception:
                # mimic VBA IsDate = False
                pass

    return np.array(times), np.array(rainfall)


# =====================================================
# VBA-STYLE AMS COMPUTATION
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    """
    Optimized VBA-compatible AMS
    - O(N) time
    - Preserves row order
    - Year assigned at END of window
    """
    window = int(duration_min / interval_min)
    n = len(rain)

    # cumulative sum
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}

    for i in range(window - 1, n):
        # rolling sum using prefix sum
        window_sum = cumsum[i + 1] - cumsum[i + 1 - window]
        year = times[i].year  # VBA behavior

        if year not in ams or window_sum > ams[year]:
            ams[year] = window_sum

    return ams


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🌧️ Annual Maximum Series (AMS)")
st.caption("VBA-compatible rolling-window AMS generator")

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

if not files:
    st.info("Upload rainfall data files to generate AMS.")
    st.stop()

# =====================================================
# PROCESS DATA
# =====================================================
times, rain = read_rainfall_vba_style(files)

if len(rain) == 0:
    st.error("No valid rainfall records found.")
    st.stop()

# =====================================================
# COMPUTE AMS
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
