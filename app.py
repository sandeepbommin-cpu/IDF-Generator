import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF / ABM Generator", layout="wide")

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
    vals = []
    for v in text.split(","):
        v = v.strip()
        if v.isdigit():
            vals.append(int(v))
    return sorted(set(vals))

def sort_files_by_numeric_suffix(files):
    def extract_index(f):
        m = re.search(r'_(\d+)', f.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =====================================================
# READ RAINFALL DATA (UPLOAD ONLY – ROW ORDER PRESERVED)
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
# AMS (VBA‑EQUIVALENT, PYTHON SAFE)
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
# DISTRIBUTIONS
# =====================================================
def gumbel_excel_q(x, T):
    mean = x.mean()
    s = x.std(ddof=1)
    yT = -np.log(np.log(T / (T - 1.0)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y
    return mean + KT * s

def gev_q(x, T):
    lm = distr.gev.lmom_fit(x)
    return distr.gev.ppf(1 - 1 / T, **lm)

def lp3_q(x, T):
    params = pearson3.fit(np.log(x))
    return np.exp(pearson3.ppf(1 - 1 / T, *params))

def lognormal_q(x, T):
    shape, loc, scale = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1 / T, shape, loc, scale)

# =====================================================
# TRUE ABM (DDF‑BASED WITH INTERPOLATION)
# =====================================================
def abm_from_ddf(ddf_row, duration_min, timestep_min):
    times_known = np.insert(np.array(ddf_row.index, dtype=float), 0, 0.0)
    depth_known = np.insert(np.array(ddf_row.values, dtype=float), 0, 0.0)

    t = np.arange(timestep_min, duration_min + timestep_min, timestep_min)
    cumulative = np.interp(t, times_known, depth_known)
    inc = np.diff(np.insert(cumulative, 0, 0.0))

    blocks = np.sort(inc)[::-1]
    h = np.zeros(len(inc))
    c = len(h) // 2
    h[c] = blocks[0]

    left, right = c - 1, c + 1
    for i in range(1, len(blocks)):
        if i % 2 == 1 and right < len(h):
            h[right] = blocks[i]
            right += 1
        elif left >= 0:
            h[left] = blocks[i]
            left -= 1

    return h.round(2)

# =====================================================
# UI
# =====================================================
st.title("🌧️ AMS / DDF / IDF / ABM Generator")

with st.sidebar:
    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)

    # Durations
    st.markdown("### Durations (minutes)")
    predefined_dur = [5, 10, 15, 30, 60, 120, 360, 720, 1440]
    sel_dur_pre = st.multiselect("Select from list", predefined_dur, default=[30, 60, 120])
    sel_dur_custom = parse_custom_values(
        st.text_input("Add custom durations", placeholder="e.g. 45, 90")
    )
    durations = sorted(set(sel_dur_pre + sel_dur_custom))
    st.info(f"✅ Durations used: {durations}")

    # Return periods
    st.markdown("### Return periods (years)")
    predefined_T = [2, 5, 10, 20, 30, 50, 100]
    sel_T_pre = st.multiselect("Select from list", predefined_T, default=[10])
    sel_T_custom = parse_custom_values(
        st.text_input("Add custom return periods", placeholder="e.g. 25, 75")
    )
    Tvals = sorted(set(sel_T_pre + sel_T_custom))
    st.info(f"✅ Return periods used: {Tvals}")

    distributions = st.multiselect(
        "Distributions for DDF / IDF",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    files = st.file_uploader(
        "Upload rainfall files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    st.divider()

    abm_dist = st.selectbox("ABM distribution", distributions)
    abm_dur = st.multiselect("ABM durations", durations, default=durations[:1])
    abm_T = st.multiselect("ABM return periods", Tvals, default=Tvals[:1])

    btn_ams = st.button("Compute AMS")
    btn_ddf = st.button("Compute DDF / IDF")
    btn_abm_tables = st.button("Compute ABM Tables")
    btn_abm_plots = st.button("Show ABM Hyetographs")

# =====================================================
# REQUIRE DATA
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
    AMS = {d: compute_ams_vba(times, rain, d, interval) for d in durations}
    st.session_state["AMS"] = AMS
    st.subheader("📊 AMS")
    st.dataframe(pd.DataFrame(AMS).round(2))

# =====================================================
# DDF / IDF
# =====================================================
if btn_ddf:
    if "AMS" not in st.session_state:
        st.warning("Compute AMS first.")
        st.stop()

    DDF = {}
    for dist in distributions:
        table = {}
        for d, x in st.session_state["AMS"].items():
            table[d] = [
                gumbel_excel_q(x, T) if dist == "Gumbel"
                else gev_q(x, T) if dist == "GEV"
                else lp3_q(x, T) if dist == "LP-III"
                else lognormal_q(x, T)
                for T in Tvals
            ]

        df = pd.DataFrame(table, index=Tvals).T.round(2)
        DDF[dist] = df

        st.subheader(f"📐 DDF – {dist} (mm)")
        st.dataframe(df)

        st.subheader(f"📐 IDF – {dist} (mm/hr)")
        st.dataframe(df.div(df.index.values / 60.0, axis=0).round(2))

    st.session_state["DDF"] = DDF

# =====================================================
# ABM TABLES
# =====================================================
if btn_abm_tables:
    if "DDF" not in st.session_state:
        st.warning("Compute DDF first.")
        st.stop()

    ABM = {}
    ddf_sel = st.session_state["DDF"][abm_dist]

    for d in abm_dur:
        for T in abm_T:
            h = abm_from_ddf(ddf_sel.loc[d], d, interval)
            t = np.arange(interval, d + interval, interval)

            df = pd.DataFrame({
                "Time (min)": t,
                "Incremental Rainfall (mm)": h,
                "Cumulative Rainfall (mm)": np.cumsum(h).round(2)
            })

            ABM[(d, T)] = df
            st.subheader(f"ABM Table – {d} min, T={T} yr ({abm_dist})")
            st.dataframe(df)

    st.session_state["ABM"] = ABM

# =====================================================
# ABM HYETOGRAPHS (PLOTS ONLY WHEN REQUESTED)
# =====================================================
if btn_abm_plots:
    if "ABM" not in st.session_state:
        st.warning("Compute ABM tables first.")
        st.stop()

    for (d, T), df in st.session_state["ABM"].items():
        st.subheader(f"ABM Hyetograph – {d} min, T={T} yr ({abm_dist})")
        st.bar_chart(
            df.set_index("Time (min)")["Incremental Rainfall (mm)"],
            height=300
        )
