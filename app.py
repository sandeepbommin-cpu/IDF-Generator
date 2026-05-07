import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="AMS / DDF / IDF / HEC‑HMS Frequency Storm",
    layout="wide"
)

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
# READ RAINFALL DATA (UPLOAD ONLY)
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
# AMS (VBA‑EQUIVALENT LOGIC)
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
        else:
            ams[yr] = max(ams[yr], wsum)

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
    return distr.gev.ppf(1 - 1 / T, **distr.gev.lmom_fit(x))

def lp3_q(x, T):
    params = pearson3.fit(np.log(x))
    return np.exp(pearson3.ppf(1 - 1 / T, *params))

def lognormal_q(x, T):
    shape, loc, scale = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1 / T, shape, loc, scale)

# =====================================================
# ✅ HEC‑HMS FREQUENCY STORM (NESTED ABM)
# =====================================================
def hms_frequency_storm(ddf_row, timestep_min, peak_position=0.5):

    durations = np.array(sorted(ddf_row.index), dtype=int)
    depths = ddf_row.loc[durations].values

    inc_depths = np.diff(np.insert(depths, 0, 0.0))
    inc_durations = np.diff(np.insert(durations, 0, 0))

    blocks = []
    for d_inc, t_inc in zip(inc_depths, inc_durations):
        n = int(t_inc / timestep_min)
        blocks.extend([d_inc / n] * n)

    blocks = np.array(blocks)
    n_blocks = len(blocks)
    peak_idx = int(n_blocks * peak_position)

    hyeto = np.zeros(n_blocks)
    hyeto[peak_idx] = blocks[0]

    left, right = peak_idx - 1, peak_idx + 1
    i = 1
    while i < n_blocks:
        if right < n_blocks:
            hyeto[right] = blocks[i]
            i += 1
            right += 1
        if i < n_blocks and left >= 0:
            hyeto[left] = blocks[i]
            i += 1
            left -= 1

    time = np.arange(timestep_min, timestep_min * (n_blocks + 1), timestep_min)

    df = pd.DataFrame({
        "Time (min)": time,
        "Incremental Rainfall (mm)": np.round(hyeto, 3)
    })
    df["Cumulative Rainfall (mm)"] = df["Incremental Rainfall (mm)"].cumsum().round(3)

    return df

# =====================================================
# UI
# =====================================================
st.title("🌧️ AMS / DDF / IDF / HEC‑HMS Frequency Storm")

with st.sidebar:

    interval = st.number_input("Data interval / timestep (minutes)", 1, value=6)

    st.markdown("### Durations (minutes)")
    durations = st.multiselect(
        "Select durations",
        [6, 30, 60, 120, 360, 720, 1440],
        default=[6, 30, 60, 120, 360, 720, 1440]
    )

    st.markdown("### Return periods (years)")
    selected_T = st.multiselect(
        "Select return periods",
        [2, 5, 10, 20, 30, 50, 100],
        default=[10]
    )

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    files = st.file_uploader(
        "Upload rainfall files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    st.divider()

    # ===== ABM CONTROLS =====
    st.markdown("## 🌧️ HEC‑HMS ABM")

    abm_distribution = st.selectbox("ABM Distribution", distributions)
    abm_return_periods = st.multiselect(
        "ABM Return periods",
        selected_T,
        default=selected_T[:1]
    )
    abm_durations = st.multiselect(
        "ABM Storm durations (min)",
        durations,
        default=[max(durations)]
    )

    btn_ams = st.button("Compute AMS")
    btn_ddf = st.button("Compute DDF / IDF")
    btn_abm = st.button("Compute ABM Tables")

    show_abm_plots = st.checkbox("Show ABM Hyetographs")

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
    st.subheader("📊 Annual Maximum Series (AMS)")
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
                else gev_q(x, T)
                if dist == "GEV"
                else lp3_q(x, T)
                if dist == "LP-III"
                else lognormal_q(x, T)
                for T in selected_T
            ]

        df = pd.DataFrame(table, index=selected_T).T.round(2)
        DDF[dist] = df

        st.subheader(f"📐 DDF – {dist}")
        st.dataframe(df)

    st.session_state["DDF"] = DDF

# =====================================================
# ✅ ABM TABLES (HMS)
# =====================================================
if btn_abm:

    if "DDF" not in st.session_state:
        st.warning("Compute DDF first.")
        st.stop()

    ABM_RESULTS = {}
    ddf_df = st.session_state["DDF"][abm_distribution]

    for T in abm_return_periods:
        for D in abm_durations:
            ddf_row = ddf_df.loc[ddf_df.index <= D, T]
            storm = hms_frequency_storm(ddf_row, interval)

            key = (abm_distribution, T, D)
            ABM_RESULTS[key] = storm

            st.subheader(f"ABM Table – {abm_distribution}, T={T} yr, D={D} min")
            st.dataframe(storm)

    st.session_state["ABM_RESULTS"] = ABM_RESULTS

# =====================================================
# ✅ ABM HYETOGRAPHS (ONLY WHEN ASKED)
# =====================================================
if show_abm_plots:

    if "ABM_RESULTS" not in st.session_state:
        st.warning("Compute ABM tables first.")
        st.stop()

    for (dist, T, D), df in st.session_state["ABM_RESULTS"].items():

        st.subheader(f"ABM Hyetograph – {dist}, T={T} yr, D={D} min")

        st.bar_chart(
            df.set_index("Time (min)")["Incremental Rainfall (mm)"],
            height=300
        )
