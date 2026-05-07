import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="DDF / IDF Generator (VBA Matched)", layout="wide")

# =====================================================
# CONSTANTS (Excel / VBA Gumbel)
# =====================================================
EULER_GAMMA = 0.5772156649
SIGMA_Y = 1.28255   # std dev of reduced variate

# =====================================================
# HARD-CODED AMS (VERIFIED, AUTHORITATIVE)
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
# VBA / Excel GUMBEL (FREQUENCY-FACTOR FORM)
# =====================================================
def gumbel_excel_q(x, T):
    """
    EXACT Excel/VBA Gumbel:
      X_T = mean + K_T * stdev
      K_T = (y_T - ybar) / sigma_y
      y_T = -ln( ln( T / (T-1) ) )
    """
    x = np.asarray(x, dtype=float)

    xbar = x.mean()                # AVERAGE
    s = x.std(ddof=1)              # STDEV.S

    yT = -np.log(np.log(T / (T - 1.0)))
    KT = (yT - EULER_GAMMA) / SIGMA_Y

    return xbar + KT * s

# =====================================================
# UI
# =====================================================
st.title("🌧️ DDF / IDF Generator")
st.caption("Gumbel distribution matched exactly to Excel/VBA")

with st.sidebar:
    if "return_periods" not in st.session_state:
        st.session_state.return_periods = [2, 5, 10, 20, 30, 50, 100]

    selected_T = st.multiselect(
        "Return periods (years)",
        st.session_state.return_periods,
        default=[2, 10, 30, 50, 100]
    )

    custom_T = st.text_input(
        "Add custom return periods",
        placeholder="e.g. 25, 75"
    )

    if custom_T:
        for v in custom_T.split(","):
            v = v.strip()
            if v.isdigit():
                vi = int(v)
                if vi not in st.session_state.return_periods:
                    st.session_state.return_periods.append(vi)
        st.session_state.return_periods = sorted(st.session_state.return_periods)

    run = st.button("📐 Compute DDF & IDF")

# =====================================================
# DDF / IDF
# =====================================================
if run and selected_T:

    st.subheader("📐 DDF & IDF – Gumbel (Excel/VBA)")

    ddf = {}
    for d in DURATIONS:
        x = np.array(AMS_DATA[d], dtype=float)
        ddf[d] = [gumbel_excel_q(x, T) for T in selected_T]

    ddf_df = pd.DataFrame(ddf, index=selected_T).T
    ddf_df.index.name = "Duration (min)"

    st.markdown("**Rainfall Depth (mm)**")
    st.dataframe(ddf_df, use_container_width=True)

    idf_df = ddf_df.div(ddf_df.index.values / 60.0, axis=0)

    st.markdown("**Rainfall Intensity (mm/hr)**")
    st.dataframe(idf_df, use_container_width=True)
