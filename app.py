import streamlit as st
import pandas as pd
import numpy as np
import re
import pathlib

st.set_page_config(
    page_title="AMS Generator (VBA-Compatible)",
    layout="wide"
)

DATA_DIR = pathlib.Path("data")

# =====================================================
# AUTO-SORT FILES BY NUMERIC SUFFIX (_1, _2, ...)
# =====================================================
def sort_files_by_numeric_suffix(file_paths):
    def extract_index(p):
        match = re.search(r'_(\d+)', p.name)
        return int(match.group(1)) if match else float("inf")
    return sorted(file_paths, key=extract_index)

# =====================================================
# FAST REPOSITORY-BASED RAINFALL READER
# =====================================================
@st.cache_data(show_spinner="Reading rainfall data from repository...")
def read_rainfall_from_repo():
    if not DATA_DIR.exists():
        raise FileNotFoundError("data/ directory not found in repository")

    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No rainfall files found in data/ folder")

    files = sort_files_by_numeric_suffix(files)

    all_times = []
    all_rain = []

    for f in files:
        if f.suffix.lower() == ".csv":
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

    return times, rain, files

# =====================================================
# OPTIMIZED VBA-COMPATIBLE AMS
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    window = int(duration_min / interval_min)
    n = len(rain)

    years = pd.DatetimeIndex(times).year.to_numpy()
    cumsum = np.zeros(n + 1, dtype=float)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, n):
        window_sum = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])
        if yr not in ams or window_sum > ams[yr]:
            ams[yr] = window_sum

    return ams

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🌧️ Annual Maximum Series (AMS)")
st.caption("Repository-based, VBA-compatible, button-driven AMS tool")

with st.sidebar:
    st.header("Inputs")

    interval = st.number_input(
        "Data interval (minutes)",
        min_value=1,
        value=6
    )

    base_durations = [5, 10, 15, 30, 60, 120, 360, 720, 1440]
    selected_durations = st.multiselect(
        "Select durations (minutes)",
        base_durations,
        default=[]
    )

    # ✅ Text input allows empty value safely
    manual_duration_text = st.text_input(
        "Add custom duration (minutes)",
        value="",
        placeholder="e.g. 180"
    )

    compute_button = st.button("✅ Compute AMS")

# =====================================================
# LOAD RAINFALL DATA
# =====================================================
try:
    times, rain, used_files = read_rainfall_from_repo()
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.markdown("**Rainfall files used (order):**")
for f in used_files:
    st.sidebar.write(f.name)

# =====================================================
# COMPUTE AMS (ON BUTTON CLICK)
# =====================================================
if compute_button:

    durations = list(selected_durations)

    if manual_duration_text.strip():
        if manual_duration_text.isdigit():
            durations.append(int(manual_duration_text))
        else:
            st.error("Custom duration must be a positive integer.")
            st.stop()

    if not durations:
        st.warning("Please select or enter at least one duration.")
        st.stop()

    st.subheader("📊 Annual Maximum Series (AMS)")

    ams_table = {}
    for d in sorted(set(durations)):
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

else:
    st.info("Select durations and click **Compute AMS** to generate results.")
