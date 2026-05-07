import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearson3, lognorm
from lmoments3 import distr

st.set_page_config(
    page_title="DDF / IDF Generator (AMS Frozen)",
    layout="wide"
)

# =====================================================
# CONSTANTS (Excel / VBA Gumbel)
# =====================================================
EULER_GAMMA = 0.5772156649015329
GUMBEL_SD_FACTOR = np.sqrt(6) / np.pi  # 1.28255

# =====================================================
# HARD-CODED AMS (AUTHORITATIVE, VERIFIED)
# =====================================================
AMS_DATA = {
    30:   [18.8, 19.6, 20.4, 15.6, 12.2, 16.8, 12.4, 15.8, 12.8,  7.8, 15.4, 12.2, 19.2, 16.4, 18.4],
    60:   [25.2, 36.8, 27.6, 19.0, 15.2, 17.6, 16.4, 21.8, 13.6, 11.2, 15.6, 15.8, 33.0, 18.4, 31.6],
    120:  [29.0, 56.0, 40.0, 21.2, 18.0, 21.6, 18.6, 22.6, 15.4, 11.6, 18.4, 24.59, 55.4, 18.6, 39.6],
    360:  [36.4, 60.0, 45.6, 28.8, 21.0, 39.4, 26.2, 32.2, 23.6, 19.8, 34.4, 46.01, 64.2, 36.8, 58.6],
    720:  [60.6, 95.6, 53.2, 31.2, 31.2, 39.4, 26.6, 46.2, 28.0, 22.6, 34.75, 57.67, 64.2, 43.8, 66.0],
    1440: [70.8,104.6, 64.2, 34.4, 33.4, 42.6, 28.8, 70.2, 28.0, 24.8, 48.4, 66.37, 71.0, 54.6, 94.2]
}

DURATIONS = list(AMS_DATA.keys())

# =====================================================
# DISTRIBUTIONS
# =====================================================
def gumbel_vba_q(x, T):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    s = x.std(ddof=1)           # STDEV.S
    beta = s / GUMBEL_SD_FACTOR
    alpha = mu - EULER_GAMMA * beta
    yT = -np.log(np.log(T / (T - 1.0)))
    return alpha + beta * yT

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
st.title("🌧️ DDF / IDF Generator")
st.caption("AMS frozen; Excel/VBA‑matched Gumbel implementation")

with st.sidebar:
    st.header("Return periods")

    if "return_periods" not in st.session_state:
        # ✅ 30-year added here
        st.session_state.return_periods = [2, 5, 10, 20, 30, 50, 100]

    selected_freqs = st.multiselect(
        "Return periods (years)",
        st.session_state.return_periods,
        default=[2, 10, 30, 50, 100]
    )

    custom_freq_text = st.text_input(
        "Add custom return periods",
        placeholder="e.g. 25, 75"
    )

    if custom_freq_text:
        for f in custom_freq_text.split(","):
            f = f.strip()
            if f.isdigit():
                fv = int(f)
                if fv not in st.session_state.return_periods:
                    st.session_state.return_periods.append(fv)
        st.session_state.return_periods = sorted(st.session_state.return_periods)

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    run_button = st.button("📐 Compute DDF & IDF")

# =====================================================
# DDF / IDF
# =====================================================
if run_button and distributions and selected_freqs:

    for dist in distributions:
        st.subheader(f"📐 DDF & IDF – {dist}")

        ddf = {}
        for d in DURATIONS:
            x = np.array(AMS_DATA[d], dtype=float)
            vals = []

            for T in selected_freqs:
                if dist == "Gumbel":
                    vals.append(gumbel_vba_q(x, T))
                elif dist == "GEV":
                    vals.append(gev_q(x, T))
                elif dist == "LP-III":
                    vals.append(lp3_q(x, T))
                elif dist == "Lognormal":
                    vals.append(lognormal_q(x, T))

            ddf[d] = vals

        ddf_df = pd.DataFrame(ddf, index=selected_freqs).T

        st.markdown("**Rainfall Depth (mm)**")
        st.dataframe(ddf_df, use_container_width=True)

        idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0)

        st.markdown("**Rainfall Intensity (mm/hr)**")
        st.dataframe(idf_df, use_container_width=True)
