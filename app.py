import io
import os
import csv
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_autorefresh import st_autorefresh


# =========================================================
# ADMIN KEY (اختياري – للتحكم في الإعدادات)
# =========================================================
def get_admin_key():
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        return ""


ADMIN_KEY = get_admin_key()
MODEL_VERSION = "v2.1"

st.set_page_config(page_title="Clinical–Actuarial UPI Dashboard", layout="wide")
st.title("Clinical–Actuarial UPI Dashboard – 3-Level Model + Governance")

admin_input = st.sidebar.text_input("Admin key (optional)", type="password")
IS_ADMIN = ADMIN_KEY != "" and admin_input == ADMIN_KEY

if not IS_ADMIN:
    st_autorefresh(interval=15 * 60 * 1000, key="public_refresh")
    st.sidebar.caption("Public mode (auto-refresh every 15 minutes)")
else:
    st.sidebar.caption("Admin mode active")


# =========================================================
# UPLOAD DATA
# =========================================================
uploaded = st.file_uploader("Upload Patient File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Please upload a dataset.")
    st.stop()

dataset_name = uploaded.name

ext = uploaded.name.split(".")[-1].lower()
if ext == "csv":
    data = pd.read_csv(uploaded)
else:
    data = pd.read_excel(uploaded)

required_cols = ["patient_id", "BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Optional columns
has_region = "region" in data.columns
has_provider = "provider_name" in data.columns
has_period = "period" in data.columns
has_drg = "drg_group" in data.columns
has_claims = "claims_amount" in data.columns
has_premium = "premium_amount" in data.columns


# =========================================================
# NORMALIZATION & WEIGHTS
# =========================================================
st.sidebar.markdown("### Normalization")

norm_method = st.sidebar.selectbox(
    "Normalization method",
    ["MinMax (0–100)", "Z-score (mean±SD)", "None"],
    index=0,
)


def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if norm_method == "None":
        return s
    if norm_method == "MinMax (0–100)":
        return 100 * (s - s.min()) / (s.max() - s.min() + 1e-9)
    if norm_method == "Z-score (mean±SD)":
        z = (s - s.mean()) / (s.std() + 1e-9)
        return 50 + 10 * z
    return s


for col in ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]:
    data[col] = normalize(data[col])


st.sidebar.markdown("### UPI Weights")
wBAS = st.sidebar.slider("BAS weight", 0.0, 1.0, 0.25, 0.01)
wCRS = st.sidebar.slider("CRS weight", 0.0, 1.0, 0.25, 0.01)
wCARS = st.sidebar.slider("CARS weight", 0.0, 1.0, 0.25, 0.01)
wPEN = st.sidebar.slider("Provider Penalty weight", 0.0, 1.0, 0.15, 0.01)
wFEI = st.sidebar.slider("FEI weight", 0.0, 1.0, 0.10, 0.01)

total_w = wBAS + wCRS + wCARS + wPEN + wFEI
if abs(total_w - 1.0) > 0.02:
    st.sidebar.warning(f"Total weights = {total_w:.2f}. Ideal total = 1.00")

st.sidebar.markdown("### Risk thresholds")
high_thr = st.sidebar.slider("High-risk threshold", 50, 95, 80, 1)
med_thr = st.sidebar.slider("Medium-risk threshold", 40, 89, 60, 1)


# =========================================================
# CALCULATIONS
# =========================================================
def calc_provider_penalty(row):
    return 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS"])


def calc_upi(row):
    return (
        wBAS * row["BAS"]
        + wCRS * row["CRS"]
        + wCARS * row["CARS"]
        + wPEN * row["provider_penalty"]
        + wFEI * row["FEI"]
    )


def classify_risk(upi):
    if upi >= high_thr:
        return "High Risk"
    if upi >= med_thr:
        return "Medium Risk"
    return "Low Risk"


data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)
data["UPI"] = data.apply(calc_upi, axis=1)
data["risk_level"] = data["UPI"].apply(classify_risk)


# =========================================================
# LOGGING SYSTEM
# =========================================================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "upi_runs_log.csv")


def log_run(df, dataset_name):
    os.makedirs(LOG_DIR, exist_ok=True)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(df)
    high = int((df["risk_level"] == "High Risk").sum())
    pct = (100 * high / total) if total > 0 else 0
    avg_upi = df["UPI"].mean() if total > 0 else 0

    weights = f"BAS={wBAS:.2f};CRS={wCRS:.2f};CARS={wCARS:.2f};PEN={wPEN:.2f};FEI={wFEI:.2f}"
    thresholds = f"high={high_thr};medium={med_thr}"

    header = [
        "timestamp", "model_version", "dataset_name",
        "total_patients", "high_risk_patients", "high_risk_pct",
        "avg_upi", "norm_method", "weights", "thresholds"
    ]

    row = [
        now, MODEL_VERSION, dataset_name,
        total, high, f"{pct:.2f}",
        f"{avg_upi:.2f}", norm_method, weights, thresholds
    ]

    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


# =========================================================
# TABS (4 Tabs)
# =========================================================
tab_strat, tab_fac, tab_patient, tab_logs = st.tabs([
    "Level 1 – Strategic Executive Dashboard",
    "Level 2 – Facility Performance",
    "Level 3 – Patient Risk Panel",
    "Model Governance / Logs",
])


# ---------------------------------------------------------
# LEVEL 1 – STRATEGIC DASHBOARD
# ---------------------------------------------------------
with tab_strat:
    st.subheader("Level 1 – Strategic Executive Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", len(data))
    c2.metric("High-Risk", int((data["risk_level"] == "High Risk").sum()))
    c3.metric("Average UPI", f"{data['UPI'].mean():.1f}")

    if st.button("Log this run (append to logs/upi_runs_log.csv)"):
        log_run(data, dataset_name)
        st.success("Run logged successfully")

    st.markdown("---")

    # 1) UPI Trend
    st.markdown("### 1) UPI Trend")
    if has_period:
        trend = data.groupby("period")["UPI"].mean().reset_index()
        fig = px.line(trend, x="period", y="UPI", markers=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add 'period' column to enable UPI Trend")

    # 2) Loss Ratio vs UPI
    st.markdown("### 2) Loss Ratio vs UPI")
    if has_claims and has_premium and has_period:
        dfc = data.groupby("period").agg(
            avg_upi=("UPI", "mean"),
            claims=("claims_amount", "sum"),
            premium=("premium_amount", "sum"),
        ).reset_index()
        dfc["loss_ratio"] = 100 * dfc["claims"] / dfc["premium"].replace(0, np.nan)

        fig = px.line(
            dfc, x="period", y=["avg_upi", "loss_ratio"],
            title="Loss Ratio vs UPI", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add claims_amount + premium_amount + period to enable this chart")

    # 3) Regional risk
    st.markdown("### 3) Average UPI by Region")
    if has_region:
        reg = data.groupby("region")["UPI"].mean().reset_index()
        st.plotly_chart(px.bar(reg, x="region", y="UPI", template="plotly_white"), use_container_width=True)
    else:
        st.info("Add 'region' column to enable Region View")

    # 4) Provider Ranking
    st.markdown("### 4) Top 10 Providers by UPI")
    if has_provider:
        prov = (
            data.groupby("provider_name")["UPI"]
            .mean()
            .reset_index()
            .sort_values("UPI", ascending=False)
            .head(10)
        )
        st.plotly_chart(px.bar(prov, x="provider_name", y="UPI", template="plotly_white"))
        st.dataframe(prov, use_container_width=True)
    else:
        st.info("Add 'provider_name' column to enable provider ranking")


# ---------------------------------------------------------
# LEVEL 2 – FACILITY PERFORMANCE
# ---------------------------------------------------------
with tab_fac:
    st.subheader("Level 2 – Facility Performance Dashboard")

    if not has_provider:
        st.info("Add 'provider_name' column to use this dashboard")
    else:
        providers = sorted(data["provider_name"].unique())
        selected_provider = st.selectbox("Select provider", providers)

        dfp = data[data["provider_name"] == selected_provider]

        c1, c2, c3 = st.columns(3)
        c1.metric("Patients", len(dfp))
        c2.metric("High-Risk", int((dfp["risk_level"] == "High Risk").sum()))
        c3.metric("Avg UPI", f"{dfp['UPI'].mean():.1f}")

        st.markdown("---")

        # Risk Composition
        comp = dfp[["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]].mean().reset_index()
        comp.columns = ["Score", "Value"]
        st.plotly_chart(px.bar(comp, x="Score", y="Value", template="plotly_white"))

        # UPI vs Network
        c4, c5 = st.columns(2)
        c4.metric("Provider Avg UPI", f"{dfp['UPI'].mean():.1f}")
        c5.metric("Network Avg UPI", f"{data['UPI'].mean():.1f}")

        # DRG Shift
        st.markdown("### DRG Risk Shift")
        if has_drg and has_period:
            drg = dfp.groupby(["period", "drg_group"])["UPI"].mean().reset_index()
            st.plotly_chart(px.line(drg, x="period", y="UPI", color="drg_group", markers=True, template="plotly_white"))
        else:
            st.info("Need drg_group + period")

        # PCS Trend
        st.markdown("### PCS Trend")
        if has_period:
            pcs = dfp.groupby("period")["PCS"].mean().reset_index()
            st.plotly_chart(px.line(pcs, x="period", y="PCS", markers=True, template="plotly_white"))
        else:
            st.info("Need 'period'")


# ---------------------------------------------------------
# LEVEL 3 – PATIENT PANEL
# ---------------------------------------------------------
with tab_patient:
    st.subheader("Level 3 – Patient Risk Panel")

    pts = sorted(data["patient_id"].unique())
    selected_pt = st.selectbox("Select patient", pts)

    df_pt = data[data["patient_id"] == selected_pt]
    if has_period:
        df_pt = df_pt.sort_values("period")

    latest = df_pt.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("UPI", f"{latest['UPI']:.1f}")
    c2.metric("Risk Level", latest["risk_level"])
    c3.metric("Provider", latest["provider_name"] if has_provider else "N/A")

    st.markdown("---")

    drivers = pd.DataFrame({
        "Score": ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"],
        "Value": [
            latest["BAS"], latest["CRS"], latest["CARS"],
            latest["PCS"], latest["PPS"], latest["FEI"]
        ]
    })
    st.plotly_chart(px.bar(drivers, x="Score", y="Value", template="plotly_white"))

    c4, c5 = st.columns(2)
    c4.metric("PCS", f"{latest['PCS']:.1f}")
    c5.metric("Provider Penalty", f"{latest['provider_penalty']:.1f}")

    st.markdown("### Last 3 periods")
    if has_period and len(df_pt) >= 3:
        last3 = df_pt.tail(3)
        st.plotly_chart(px.line(last3, x="period", y="UPI", markers=True, template="plotly_white"))
    else:
        st.info("Not enough period history")


# ---------------------------------------------------------
# MODEL GOVERNANCE / LOGS
# ---------------------------------------------------------
with tab_logs:
    st.subheader("Model Governance – Run Logs")

    if not os.path.isfile(LOG_FILE):
        st.info("No logs found. Go to Level 1 and click 'Log this run' at least once.")
    else:
        logs = pd.read_csv(LOG_FILE)

        # تحويل timestamp لنوع تاريخ/وقت
        if "timestamp" in logs.columns:
            logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")

        logs = logs.sort_values("timestamp")

        # ---- أدوات أعلى التبويب: Download + Clear (Admin) ----
        top_col1, top_col2 = st.columns(2)

        # زر تحميل كل الـ logs كـ CSV
        csv_buf = io.StringIO()
        logs.to_csv(csv_buf, index=False)
        top_col1.download_button(
            "Download full log (CSV)",
            data=csv_buf.getvalue(),
            file_name="upi_runs_log.csv",
            mime="text/csv",
        )

        # زر مسح الـ logs (Admin فقط)
        if IS_ADMIN:
            if top_col2.button("Clear all logs (Admin only)"):
                os.remove(LOG_FILE)
                st.warning("All logs have been cleared. New runs will create a fresh log file.")
                st.stop()
        else:
            top_col2.caption("Clear logs: Admin only")

        st.markdown("---")

        # ---- KPIs عامة عن الـ runs ----
        n_runs = len(logs)
        last_ts = logs["timestamp"].max() if "timestamp" in logs.columns else None
        last_avg = float(logs["avg_upi"].iloc[-1]) if "avg_upi" in logs.columns else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Logged Runs", n_runs)
        c2.metric("Last Run Time", str(last_ts) if last_ts is not None else "N/A")
        c3.metric("Last Avg UPI", f"{last_avg:.2f}" if last_avg is not None else "N/A")

        st.markdown("---")

        # 1) Avg UPI trend عبر الزمن
        if {"timestamp", "avg_upi"}.issubset(logs.columns):
            st.markdown("### 1) Average UPI across runs")
            fig_log_upi = px.line(
                logs,
                x="timestamp",
                y="avg_upi",
                markers=True,
                title="Average UPI per logged run",
                template="plotly_white",
            )
            st.plotly_chart(fig_log_upi, use_container_width=True)
        else:
            st.info("Missing 'timestamp' or 'avg_upi' in logs – cannot plot UPI trend.")

        # 2) High-Risk % trend
        if {"timestamp", "high_risk_pct"}.issubset(logs.columns):
            st.markdown("### 2) High-Risk % across runs")
            fig_log_hr = px.line(
                logs,
                x="timestamp",
                y="high_risk_pct",
                markers=True,
                title="High-Risk % per logged run",
                template="plotly_white",
            )
            st.plotly_chart(fig_log_hr, use_container_width=True)
        else:
            st.info("Missing 'high_risk_pct' in logs – cannot plot High-Risk % trend.")

        # 3) Normalization methods المستخدمة
        if "norm_method" in logs.columns:
            st.markdown("### 3) Normalization methods used")
            fig_norm = px.histogram(
                logs,
                x="norm_method",
                title="Runs count by normalization method",
                template="plotly_white",
            )
            st.plotly_chart(fig_norm, use_container_width=True)

        # 4) Model versions المستخدمة
        if "model_version" in logs.columns:
            st.markdown("### 4) Model versions used")
            fig_mv = px.histogram(
                logs,
                x="model_version",
                title="Runs count by model version",
                template="plotly_white",
            )
            st.plotly_chart(fig_mv, use_container_width=True)

        # 5) جدول آخر 20 Run
        st.markdown("### 5) Latest 20 logged runs")
        st.dataframe(
            logs.sort_values("timestamp", ascending=False).head(20),
            use_container_width=True,
        )



# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>Mudather</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model " + MODEL_VERSION + " • " + dt.datetime.now().strftime("%Y-%m-%d"))

