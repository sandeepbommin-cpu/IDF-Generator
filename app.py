import streamlit as st
import pandas as pd
import numpy as np
import re
import pathlib
from scipy.stats import gumbel_r, lognorm, pearson3
from lmoments3 import distr

st.set_page_config(page_title="AMS / DDF / IDF Generator", layout="wide")

DATA_DIR = pathlib.Path("data")

# ---------- File sorting ----------
def sort_files_by_numeric_suffix(files):
    def extract_index(p):
        m = re.search(r'_(\d+)', p.name)
        return int(m.group(1)) if m else float("inf")
    return sorted(files, key=extract_index)

# ---------- Rainfall reader ----------
@st.cache_data(show_spinner="Reading rainfall data...")
def read_rainfall_from_repo():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No rainfall files found in data/")

    files = sort_files_by_numeric_suffix(files)

    all_times, all_rain = [], []

    for f in files:
        df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
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

    return times, rain, files

# ---------- AMS (VBA-compatible) ----------
def compute_ams_vba(times, rain, duration_min, interval_min):
    window = int(duration_min / interval_min)
    n = len(rain)

    years = pd.DatetimeIndex(times).year.to_numpy()
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(rain)

    ams = {}
    for i in range(window - 1, n):
        wsum = cumsum[i + 1] - cumsum[i + 1 - window]
        yr = int(years[i])

        if yr not in ams:
            ams[yr] = wsum
        elif wsum > ams[yr]:
            ams[yr] = wsum

    return ams

# ---------- Distribution quantiles ----------
def q_gumbel(x, T):
    loc, scale = gumbel_r.fit(x)
    return gumbel_r.ppf(1 - 1/T, loc, scale)

def q_gev(x, T):
    lm = distr.gev.lmom_fit(x)
    return distr.gev.ppf(1 - 1/T, **lm)

def q_lp3(x, T):
    lx = np.log(x)
    params = pearson3.fit(lx)
    return np.exp(pearson3.ppf(1 - 1/T, *params))

def q_lognormal(x, T):
    shape, loc, scale = lognorm.fit(x, floc=0)
    return lognorm.ppf(1 - 1/T, shape, loc, scale)

# ---------- UI ----------
st.title("🌧️ AMS / DDF / IDF Generator")

with st.sidebar:
    interval = st.number_input("Data interval (minutes)", min_value=1, value=6)

    base_durations = [5,10,15,30,60,120,360,720,1440]
    selected_durations = st.multiselect("Durations (minutes)", base_durations)

    duration_text = st.text_input("Add custom duration (minutes)")

    default_freqs = [2,5,10,20,50,100]
    selected_freqs = st.multiselect("Return periods (years)", default_freqs)

    freq_text = st.text_input("Add custom return periods (comma-separated)")

    distributions = st.multiselect(
        "Distributions",
        ["Gumbel", "GEV", "LP-III", "Lognormal"],
        default=["Gumbel"]
    )

    compute_ams = st.button("✅ Compute AMS")
    compute_ddf = st.button("📐 Compute DDF & IDF")

# ---------- Load data ----------
times, rain, used_files = read_rainfall_from_repo()
st.sidebar.markdown("**Rainfall files used:**")
for f in used_files:
    st.sidebar.write(f.name)

# ---------- AMS ----------
if compute_ams:
    durations = list(selected_durations)
    if duration_text.isdigit():
        durations.append(int(duration_text))

    if not durations:
        st.warning("Select at least one duration.")
        st.stop()

    ams_tbl = {}
    for d in sorted(set(durations)):
        ams_tbl[f"{d} min"] = pd.Series(compute_ams_vba(times, rain, d, interval))

    ams_df = pd.DataFrame(ams_tbl).sort_index()
    st.subheader("📊 Annual Maximum Series (AMS)")
    st.dataframe(ams_df, use_container_width=True)

# ---------- DDF & IDF ----------
if compute_ddf:
    durations = list(selected_durations)
    if duration_text.isdigit():
        durations.append(int(duration_text))

    freqs = list(selected_freqs)
    if freq_text:
        for f in freq_text.split(","):
            if f.strip().isdigit():
                freqs.append(int(f.strip()))

    if not durations or not freqs or not distributions:
        st.warning("Select durations, frequencies and distributions.")
        st.stop()

    durations = sorted(set(durations))
    freqs = sorted(set(freqs))

    for dist in distributions:
        st.subheader(f"DDF & IDF – {dist}")
        ddf = {}

        for d in durations:
            x = np.array(list(compute_ams_vba(times, rain, d, interval).values()))
            vals = []

            for T in freqs:
                if dist == "Gumbel":
                    vals.append(q_gumbel(x, T))
                elif dist == "GEV":
                    vals.append(q_gev(x, T))
                elif dist == "LP-III":
                    vals.append(q_lp3(x, T))
                elif dist == "Lognormal":
                    vals.append(q_lognormal(x, T))

            ddf[d] = vals

        ddf_df = pd.DataFrame(ddf, index=freqs).T
        st.markdown("**Rainfall Depth (mm)**")
        st.dataframe(ddf_df, use_container_width=True)

        idf_df = ddf_df.div(ddf_df.index.values / 60, axis=0)
        st.markdown("**Rainfall Intensity (mm/hr)**")
        st.dataframe(idf_df, use_container_width=True)
