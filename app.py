import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gumbel_r, lognorm, pearson3
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF Generator", layout="wide")

# =========================================================
# VBA‑STYLE RAINFALL READER
# =========================================================
def read_rainfall_vba_style(files):
    records_time = []
    records_rain = []

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
                    records_time.append(t)
                    records_rain.append(r)
                except:
                    pass

    return np.array(records_time), np.array(records_rain)


# =========================================================
# VBA‑STYLE AMS COMPUTATION
# =========================================================
def compute_ams_vba(times, rain, duration_min, interval_min):
    w = int(duration_min / interval_min)
    ams = {}

    for i in range(w - 1, len(rain)):
        window_sum = rain[i - w + 1 : i + 1].sum()
        yr = times[i].year   # year at END of window (VBA behavior)

        if yr not in ams or window_sum > amsams[yr] = window_sum

    return ams


# =========================================================
# DISTRIBUTIONS
# =========================================================
def gumbel_fit(x, T):
    params = gumbel_r.fit(x)
    return gumbel_r.ppf(1 - 1 / T, *params)

def gev_fit(x, T):
    lm = distr.gev.lmom_fit(x)
    return distr.gev.ppf(1 - 1 / T, **lm)

def lp3_fit(x, T):
    lx = np.log(x)
    params = pearson3.fit(lx)
    return np.exp(pearson3.ppf(1 - 1 / T, *params))

def lognormal_fit(x, T):
    params = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1 / T, *params)


# =========================================================
# UI
# =========================================================
st.title("🌧️ AMS – DDF – IDF Generator (VBA‑Compatible)")

with st.sidebar:
    st.header("Control Inputs")
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
        "Upload rainfall data files (CSV / XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

if not files:
    st.info("Upload rainfall files to begin.")
    st.stop()


# =========================================================
# DATA INGESTION
# =========================================================
times, rain = read_rainfall_vba_style(files)

if len(rain) == 0:
    st.error("No valid rainfall data found.")
    st.stop()


# =========================================================
# AMS TABLE
# =========================================================
st.subheader("📊 Annual Maximum Series (AMS)")

ams_table = {}

for d in durations:
    ams_dict = compute_ams_vba(times, rain, d, interval)
    ams_table[f"{d} min"] = pd.Series(ams_dict)

ams_df = pd.DataFrame(ams_table)
ams_df.index.name = "Year"

st.dataframe(ams_df, use_container_width=True)
st.download_button("Download AMS CSV", ams_df.to_csv().encode(), "AMS.csv")


# =========================================================
# DDF / IDF TABLES
# =========================================================
for dist in distributions:
    st.subheader(f"📐 DDF & IDF – {dist}")

    # ---------- DDF ----------
    ddf = {}

    for d in durations:
        ams_vals = compute_ams_vba(times, rain, d, interval)
        x = np.array(list(ams_vals.values()))
        vals = []

        for T in return_periods:
            if dist == "Gumbel":
                vals.append(gumbel_fit(x, T))
            elif dist == "GEV":
                vals.append(gev_fit(x, T))
            elif dist == "LP-III":
                vals.append(lp3_fit(x, T))
            elif dist == "Lognormal":
                vals.append(lognormal_fit(x, T))

        ddf[d] = vals

    ddf_df = pd.DataFrame(ddf, index=return_periods).T
    ddf_df.index.name = "Duration (min)"
    ddf_df.columns = [f"T={t}" for t in return_periods]

    st.markdown("**Rainfall Depth (mm)**")
    st.dataframe(ddf_df, use_container_width=True)
    st.download_button(
        f"Download DDF ({dist})",
        ddf_df.to_csv().encode(),
        f"DDF_{dist}.csv",
    )

    # ---------- IDF ----------
    idf_df = ddf_df.div(ddf_df.index.values / 60, axis=0)

    st.markdown("**Rainfall Intensity (mm/hr)**")
    st.dataframe(idf_df, use_container_width=True)
    st.download_button(
        f"Download IDF ({dist})",
        idf_df.to_csv().encode(),
        f"IDF_{dist}.csv",
    )
