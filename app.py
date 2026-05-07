import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearson3, lognorm
from lmoments3 import distr
import re

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="AMS / DDF / IDF / ABM",
    layout="wide"
)

st.title("🌧️ AMS / DDF / IDF / ABM")

# =====================================================
# CONSTANTS
# =====================================================
EULER_GAMMA = 0.5772156649
SIGMA_Y = 1.28255

# =====================================================
# HELPERS
# =====================================================
def parse_custom_values(text):
    if not text.strip():
        return []
    return sorted({int(v.strip()) for v in text.split(",") if v.strip().isdigit()})

def sort_files_by_numeric_suffix(files):
    def extract_index(f):
        m = re.search(r'_(\d+)', f.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =====================================================
# READ RAINFALL DATA
# =====================================================
@st.cache_data(show_spinner="Reading rainfall data...")
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
# AMS
# =====================================================
def compute_ams_vba(times, rain, duration_min, timestep):
    window = int(duration_min / timestep)
    years = pd.DatetimeIndex(times).year.to_numpy()
    cumsum = np.zeros(len(rain) + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, len(rain)):
        val = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])
        ams[yr] = max(ams.get(yr, 0), val)

    return np.array(list(ams.values()), dtype=float)

# =====================================================
# DISTRIBUTIONS
# =====================================================
def gumbel_excel_q(x, T):
    mean = x.mean()
    s = x.std(ddof=1)
    yT = -np.log(np.log(T / (T - 1)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y
    return mean + KT * s

def gev_q(x, T):
    return distr.gev.ppf(1 - 1 / T, **distr.gev.lmom_fit(x))

def lp3_q(x, T):
    params = pearson3.fit(np.log(x))
    return np.exp(pearson3.ppf(1 - 1 / T, *params))

def lognormal_q(x, T):
    shape, loc, scale = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1 / T, shape, loc, scale)

# =====================================================
# PURE ABM
# =====================================================
def pure_abm(ddf_row, duration_min, timestep):
    time_min = np.arange(timestep, duration_min + timestep, timestep)
    cum = np.interp(time_min, ddf_row.index.values, ddf_row.values)
    inc = np.diff(np.insert(cum, 0, 0))
    sorted_inc = np.sort(inc)[::-1]

    h = np.zeros(len(sorted_inc))
    c = len(h) // 2
    h[c] = sorted_inc[0]

    L, R = c - 1, c + 1
    for i in range(1, len(sorted_inc)):
        if i % 2 == 1 and R < len(h):
            h[R] = sorted_inc[i]
            R += 1
        elif L >= 0:
            h[L] = sorted_inc[i]
            L -= 1

    return pd.DataFrame({
        "Time (min)": time_min,
        "Rainfall Increment (mm)": np.round(h, 3),
        "Cumulative Rainfall (mm)": np.cumsum(h).round(3)
    })

# =====================================================
# HEC‑HMS ABM
# =====================================================
def hms_frequency_storm(ddf_row, timestep):
    durations = np.array(sorted(ddf_row.index))
    depths = ddf_row.loc[durations].values

    inc = np.diff(np.insert(depths, 0, 0))
    dur = np.diff(np.insert(durations, 0, 0))

    blocks = []
    for p, d in zip(inc, dur):
        n = int(d / timestep)
        blocks.extend([p / n] * n)

    blocks = np.array(blocks)
    n = len(blocks)
    c = n // 2

    h = np.zeros(n)
    h[c] = blocks[0]

    L, R = c - 1, c + 1
    i = 1
    while i < n:
        if R < n:
            h[R] = blocks[i]
            i += 1
            R += 1
        if i < n and L >= 0:
            h[L] = blocks[i]
            i += 1
            L -= 1

    time_min = np.arange(timestep, timestep * (n + 1), timestep)
    return pd.DataFrame({
        "Time (min)": time_min,
        "Rainfall Increment (mm)": np.round(h, 3),
        "Cumulative Rainfall (mm)": np.cumsum(h).round(3)
    })

# =====================================================
# SIDEBAR – CONTROL PANEL
# =====================================================
with st.sidebar:

    st.markdown("### Upload rainfall files")
    files = st.file_uploader(
        "Select CSV or Excel rainfall files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    timestep = st.number_input("Data interval (minutes)", min_value=1, value=6)

    st.markdown("### AMS – Select durations (min)")
    base_dur = [6, 30, 60, 120, 360, 720, 1440]
    d1 = st.multiselect("Predefined durations", base_dur, default=[30, 60, 120])
    d2 = parse_custom_values(st.text_input("Add custom durations"))
    durations = sorted(set(d1 + d2))
    btn_ams = st.button("Compute AMS")

    st.markdown("### DDF / IDF – Select return periods (years)")
    r1 = st.multiselect("Predefined return periods", [2, 5, 10, 20, 30, 50, 100], default=[10])
    r2 = parse_custom_values(st.text_input("Add custom return periods"))
    Tvals = sorted(set(r1 + r2))

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP‑III", "Lognormal"],
        default=["Gumbel"]
    )
    btn_ddf = st.button("Compute DDF / IDF")

    st.markdown("### ABM Generation")
    abm_method = st.radio("ABM Method", ["Pure ABM", "HEC‑HMS"])
    abm_dist = st.selectbox("ABM Distribution", distributions)
    abm_T = st.multiselect("ABM Return periods (years)", Tvals, default=Tvals[:1] if Tvals else [])
    abm_D = st.multiselect("ABM Durations (min)", durations, default=[max(durations)] if durations else [])
    btn_abm = st.button("Compute ABM Tables")
    btn_plot = st.button("Show ABM Hyetographs")

# =====================================================
# MAIN
# =====================================================
if files:
    files = sort_files_by_numeric_suffix(files)
    times, rain = read_rainfall_from_upload(files)

# AMS
if btn_ams and files:
    AMS = {d: compute_ams_vba(times, rain, d, timestep) for d in durations}
    st.session_state["AMS"] = AMS
    st.subheader("📊 Annual Maximum Series (AMS) [mm]")
    st.dataframe(pd.DataFrame(AMS).round(2))

# DDF / IDF
if btn_ddf and "AMS" in st.session_state:
    DDF = {}
    for dist in distributions:
        tbl = {
            d: [
                gumbel_excel_q(st.session_state["AMS"][d], T) if dist == "Gumbel"
                else gev_q(st.session_state["AMS"][d], T) if dist == "GEV"
                else lp3_q(st.session_state["AMS"][d], T) if dist == "LP‑III"
                else lognormal_q(st.session_state["AMS"][d], T)
                for T in Tvals
            ]
            for d in durations
        }

        ddf = pd.DataFrame(tbl, index=Tvals).T.round(2)
        DDF[dist] = ddf
        st.subheader(f"📐 DDF – {dist} [mm]")
        st.dataframe(ddf)

        idf = ddf.div(ddf.index.values / 60.0, axis=0).round(2)
        st.subheader(f"📐 IDF – {dist} [mm/hr]")
        st.dataframe(idf)

    st.session_state["DDF"] = DDF

# ABM Tables
if btn_abm and "DDF" in st.session_state:
    ABM = {}
    ddf = st.session_state["DDF"][abm_dist]
    for T in abm_T:
        for D in abm_D:
            row = ddf.loc[ddf.index <= D, T]
            storm = (
                pure_abm(row, D, timestep)
                if abm_method == "Pure ABM"
                else hms_frequency_storm(row, timestep)
            )
            ABM[(abm_method, T, D)] = storm
            st.subheader(f"{abm_method} – T={T} yr, D={D} min")
            st.dataframe(storm)
    st.session_state["ABM"] = ABM

# ABM Plots
if btn_plot and "ABM" in st.session_state:
    for (m, T, D), df in st.session_state["ABM"].items():
        st.subheader(f"{m} Hyetograph – T={T} yr, D={D} min")
        st.bar_chart(df.set_index("Time (min)")["Rainfall Increment (mm)"])
