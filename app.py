import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gumbel_r, lognorm, pearson3
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF Generator (VBA-Compatible)",
                   layout="wide")

# =====================================================
# DATA INGESTION (VBA STYLE)
# =====================================================
def read_rainfall_vba_style(files):
    times = []
    rain = []

    for f in files:
        if f.name.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        df.columns = df.columns.str.lower()
        time_col = [c for c in df.columns if "time" in c or "date" in c][0]
        rain_col = [c for c in df.columns if "rain" in c][0]

        for _, row in df.iterrows():
            if pd.notna(row[time_col]):
                try:
                    t = pd.to_datetime(row[time_col])
                    r = float(row[rain_col])
                    times.append(t)
                    rain.append(r)
                except Exception:
                    pass

    return np.array(times), np.array(rain)


# =====================================================
# AMS (EXACT VBA LOGIC)
# =====================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    window = int(duration_min / interval_min)
    ams = {}

    for i in range(window - 1, len(rain)):
        window_sum = rain[i - window + 1 : i + 1].sum()
        year = times[i].year  # year at END of window (VBA behaviour)

        if year not in ams:
            ams[year] = window_sum
        else:
            if window_sum > ams[year]:
                ams[year] = window_sum

    return ams


# =====================================================
# DISTRIBUTIONS
# =====================================================
def gumbel_q(x, T):
    loc, scale = gumbel_r.fit(x)
    return gumbel_r.ppf(1 - 1 / T, loc, scale)

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
st.title("🌧️ AMS – DDF – IDF Generator (VBA-Compatible)")

with st.sidebar:
    st.header("Inputs")
    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)
    durations = st.multiselect(
        "Durations (minutes)",
        [5, 10, 15, 30, 60, 120, 360, 720, 1440],
        default=[60, 1440],
    )
    return_periods = st.multiselect(
        "Return periods (years)",
        [2, 5, 10, 20, 25, 50, 100],
        default=[2, 10, 50, 100],
    )
    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel", "GEV"],
    )
    files = st.file_uploader(
        "Upload rainfall files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

if not files:
    st.info("Upload rainfall files to proceed.")
    st.stop()

# =====================================================
# PROCESSING
# =====================================================
times, rain = read_rainfall_vba_style(files)

if len(rain) == 0:
    st.error("No valid rainfall data found.")
    st.stop()

# =====================================================
# AMS TABLE
# =====================================================
st.subheader("📊 Annual Maximum Series (AMS)")

ams_data = {}

for d in durations:
    ams = compute_ams_vba(times, rain, d, interval)
    ams_data[f"{d} min"] = pd.Series(ams)

ams_df = pd.DataFrame(ams_data)
ams_df.index.name = "Year"

st.dataframe(ams_df, use_container_width=True)
st.download_button("Download AMS CSV",
                   ams_df.to_csv().encode(),
                   "AMS.csv")

# =====================================================
# DDF / IDF
# =====================================================
for dist in distributions:
    st.subheader(f"📐 DDF & IDF – {dist}")

    ddf_rows = {}

    for d in durations:
        ams = compute_ams_vba(times, rain, d, interval)
        x = np.array(list(ams.values()), dtype=float)

        values = []
        for T in return_periods:
            if dist == "Gumbel":
                values.append(gumbel_q(x, T))
            elif dist == "GEV":
                values.append(gev_q(x, T))
            elif dist == "LP-III":
                values.append(lp3_q(x, T))
            elif dist == "Lognormal":
                values.append(lognormal_q(x, T))

        ddf_rows[d] = values

    ddf_df = pd.DataFrame(ddf_rows, index=return_periods).T
    ddf_df.index.name = "Duration (min)"
    ddf_df.columns = [f"T={t}" for t in return_periods]

    st.markdown("**Rainfall Depth (mm)**")
    st.dataframe(ddf_df, use_container_width=True)

    idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0)
    st.markdown("**Rainfall Intensity (mm/hr)**")
    st.dataframe(idf_df, use_container_width=True)
