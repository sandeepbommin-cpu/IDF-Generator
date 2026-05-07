import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF / ABM Generator", layout="wide")

# =====================================================
# CONSTANTS (Excel / VBA Gumbel)
# =====================================================
EULER_GAMMA = 0.5772156649
SIGMA_Y = 1.28255  # Std dev of reduced variate (Excel)

# =====================================================
# Helper: sort uploaded files by numeric suffix
# =====================================================
def sort_files_by_numeric_suffix(files):
    def extract_index(f):
        m = re.search(r'_(\d+)', f.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# =====================================================
# Read rainfall data (UPLOAD ONLY, VBA-compatible)
# =====================================================
@st.cache_data(show_spinner="Reading uploaded rainfall data...")
def read_rainfall_from_upload(files):
    all_times = []
    all_rain = []

    for f in files:
        df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
        df.columns = df.columns.str.lower()

        time_col = next(c for c in df.columns if "time" in c or "date" in c)
        rain_col = next(c for c in df.columns if "rain" in c)

        t = pd.to_datetime(df[time_col], errors="coerce")
        r = pd.to_numeric(df[rain_col], errors="coerce")

        mask = t.notna() & r.notna()
        all_times.append(t[mask])
        all_rain.append(r[mask])

    times = pd.concat(all_times, ignore_index=True).to_numpy()
    rain = pd.concat(all_rain, ignore_index=True).to_numpy()

    return times, rain

# =====================================================
# AMS (EXACT VBA LOGIC)
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
# Distributions
# =====================================================
def gumbel_excel_q(x, T):
    xbar = x.mean()           # AVERAGE
    s = x.std(ddof=1)         # STDEV.S
    yT = -np.log(np.log(T / (T - 1.0)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y
    return xbar + KT * s

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
# Alternating Block Method (ABM)
# =====================================================
def alternating_block_method(total_depth, duration_min, timestep_min):
    n = int(duration_min / timestep_min)

    cumulative = np.linspace(timestep_min, duration_min, n) / duration_min * total_depth
    increments = np.diff(np.insert(cumulative, 0, 0))

    sorted_blocks = np.sort(increments)[::-1]

    hyeto = np.zeros(n)
    center = n // 2
    hyeto[center] = sorted_blocks[0]

    left = center - 1
    right = center + 1

    for i in range(1, len(sorted_blocks)):
        if i % 2 == 1 and right < n:
            hyeto[right] = sorted_blocks[i]
            right += 1
        elif left >= 0:
            hyeto[left] = sorted_blocks[i]
            left -= 1

    return hyeto.round(2)

# =====================================================
# UI
# =====================================================
st.title("🌧️ AMS / DDF / IDF / ABM Generator")

with st.sidebar:
    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)

    durations = st.multiselect(
        "Durations for AMS/DDF (minutes)",
        [5, 10, 15, 30, 60, 120, 360, 720, 1440],
        default=[30, 60, 120, 360, 720, 1440]
    )

    if "return_periods" not in st.session_state:
        st.session_state.return_periods = [2, 5, 10, 20, 30, 50, 100]

    selected_T = st.multiselect(
        "Return periods",
        st.session_state.return_periods,
        default=[2, 5, 10, 20, 30, 50, 100]
    )

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    files = st.file_uploader(
        "Upload rainfall files (CSV / Excel)",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    btn_ams = st.button("✅ Compute AMS")
    btn_ddf = st.button("📐 Compute DDF / IDF")

    st.divider()

    abm_durations = st.multiselect(
        "ABM Durations (minutes)",
        [30, 60, 120, 360, 720, 1440],
        default=[60]
    )

    abm_T = st.multiselect(
        "ABM Return Periods",
        st.session_state.return_periods,
        default=[10]
    )

    btn_abm = st.button("🌧️ Compute ABM")

# =====================================================
# Require uploaded data
# =====================================================
if not files:
    st.info("Upload rainfall data files to begin.")
    st.stop()

files = sort_files_by_numeric_suffix(files)
times, rain = read_rainfall_from_upload(files)

# =====================================================
# AMS
# =====================================================
if btn_ams:
    ams_data = {}
    for d in durations:
        ams_data[d] = compute_ams_vba(times, rain, d, interval)

    st.session_state["AMS_DATA"] = ams_data

    ams_df = pd.DataFrame(
        {f"AMS_{d}min": ams_data[d] for d in durations}
    ).round(2)

    st.subheader("📊 Annual Maximum Series (AMS)")
    st.dataframe(ams_df, use_container_width=True)

# =====================================================
# DDF / IDF
# =====================================================
if btn_ddf:

    if "AMS_DATA" not in st.session_state:
        st.warning("Please compute AMS first.")
        st.stop()

    ddf_store = {}
    for dist in distributions:
        ddf = {}
        for d, x in st.session_state["AMS_DATA"].items():
            vals = []
            for T in selected_T:
                if dist == "Gumbel":
                    vals.append(gumbel_excel_q(x, T))
                elif dist == "GEV":
                    vals.append(gev_q(x, T))
                elif dist == "LP-III":
                    vals.append(lp3_q(x, T))
                elif dist == "Lognormal":
                    vals.append(lognormal_q(x, T))
            ddf[d] = vals

        ddf_df = pd.DataFrame(ddf, index=selected_T).T.round(2)
        ddf_store[dist] = ddf_df

        st.subheader(f"📐 DDF / IDF – {dist}")
        st.markdown("**Rainfall Depth (mm)**")
        st.dataframe(ddf_df, use_container_width=True)

        idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0).round(2)
        st.markdown("**Rainfall Intensity (mm/hr)**")
        st.dataframe(idf_df, use_container_width=True)

    st.session_state["DDF_DATA"] = ddf_store

# =====================================================
# ABM
# =====================================================
if btn_abm:

    if "AMS_DATA" not in st.session_state or "DDF_DATA" not in st.session_state:
        st.warning("Please compute AMS and DDF first.")
        st.stop()

    ddf_gumbel = st.session_state["DDF_DATA"].get("Gumbel")

    for d in abm_durations:
        for T in abm_T:
            depth = ddf_gumbel.loc[d, T]

            hyeto = alternating_block_method(depth, d, interval)
            time = np.arange(interval, d + interval, interval)

            df = pd.DataFrame({
                "Time (min)": time,
                "Incremental Rainfall (mm)": hyeto,
                "Cumulative Rainfall (mm)": np.cumsum(hyeto).round(2)
            })

            st.subheader(f"🌧️ ABM Hyetograph – {d} min, T={T} years")
            st.dataframe(df, use_container_width=True)
``
