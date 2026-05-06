import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gumbel_r, lognorm, pearson3, norm
from lmoments3 import distr

st.set_page_config(layout="wide", page_title="AMS / DDF / IDF Generator")

# =========================================
# Helper functions
# =========================================
def read_rainfall(files):
    dfs = []
    for f in files:
        if f.name.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        df.columns = df.columns.str.lower()
        time_col = [c for c in df.columns if "time" in c or "date" in c][0]
        rain_col = [c for c in df.columns if "rain" in c][0]

        df = df[[time_col, rain_col]]
        df.columns = ["time", "rain"]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        dfs.append(df)

    full = pd.concat(dfs).sort_values("time")
    full.set_index("time", inplace=True)
    return full


def compute_ams(df, duration, interval):
    w = int(duration / interval)
    rolling = df["rain"].rolling(w).sum()
    yrs = rolling.groupby(rolling.index.year).max().dropna()
    return yrs.values


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


# =========================================
# UI
# =========================================
st.title("🌧️ AMS – DDF – IDF Generator")

with st.sidebar:
    st.header("Input Controls")
    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)
    durations = st.multiselect("Durations (minutes)", [5,10,15,30,60,120,360,720,1440], default=[60,1440])
    return_periods = st.multiselect("Return periods (years)", [2,5,10,20,25,50,100], default=[2,10,50,100])
    dists = st.multiselect("Distributions",
                            ["Gumbel", "GEV", "LP-III", "Lognormal"],
                            default=["Gumbel", "GEV"])
    files = st.file_uploader("Upload rainfall files", type=["csv","xlsx"], accept_multiple_files=True)

if not files:
    st.info("Upload rainfall data to proceed")
    st.stop()

# =========================================
# DATA PROCESSING
# =========================================
rain = read_rainfall(files)

# =========================================
# AMS
# =========================================
st.subheader("📊 Annual Maximum Series (AMS)")

ams_table = {}
for d in durations:
    ams_table[f"{d} min"] = compute_ams(rain, d, interval)

ams_df = pd.DataFrame(ams_table)
ams_df.index.name = "Year"

st.dataframe(ams_df, use_container_width=True)
st.download_button("Download AMS CSV", ams_df.to_csv().encode(), "AMS.csv")

# =========================================
# DDF & IDF
# =========================================
for dist in dists:
    st.subheader(f"📐 DDF & IDF – {dist}")

    ddf = {}
    for d in durations:
        x = compute_ams(rain, d, interval)
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
    st.download_button(f"Download DDF ({dist})", ddf_df.to_csv().encode(), f"DDF_{dist}.csv")

    idf_df = ddf_df.div(ddf_df.index.values / 60, axis=0)
    st.markdown("**Rainfall Intensity (mm/hr)**")
    st.dataframe(idf_df, use_container_width=True)
    st.download_button(f"Download IDF ({dist})", idf_df.to_csv().encode(), f"IDF_{dist}.csv")