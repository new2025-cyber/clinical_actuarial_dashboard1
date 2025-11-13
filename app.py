import io
import os
import csv
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =====================================================================
# ADMIN & VERSION
# =====================================================================
def get_admin_key():
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        return ""

ADMIN_KEY = get_admin_key()
MODEL_VERSION = "v2.3"  # نسخة بعد دمج DRG variance + CARS

st.set_page_config(page_title="Clinical–Actuarial UPI Dashboard", layout="wide")
st.title("Clinical–Actuarial UPI Dashboard – 3 Levels + DRG + Governance")

admin_input = st.sidebar.text_input("Admin key (optional)", type="password")
IS_ADMIN = ADMIN_KEY != "" and admin_input == ADMIN_KEY

if not IS_ADMIN:
    st_autorefresh(interval=15 * 60 * 1000, key="public_refresh")
    st.sidebar.caption("Public mode (auto-refresh every 15 minutes)")
else:
    st.sidebar.caption("Admin mode active")


# =====================================================================
# UPLOAD DATA
# =====================================================================
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

# لم نعد نطلب CARS من الملف؛ سيتم حسابها في الكود
required_cols = ["patient_id", "BAS", "CRS", "PCS", "PPS", "FEI"]
missing = [c for c in required_cols if c not in data.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# أعمدة اختيارية (إن وجدت تُستخدم في التحليل)
has_region = "region" in data.columns
has_provider = "provider_name" in data.columns
has_period = "period" in data.columns
has_drg = "drg_group" in data.columns
has_claims = "claims_amount" in data.columns
has_premium = "premium_amount" in data.columns


# =====================================================================
# NORMALIZATION + WEIGHTS
# =====================================================================
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


# ---------------------------------------------------------
# DRG variance + CARS computation
# ---------------------------------------------------------
def compute_drg_variance(df: pd.DataFrame):
    """
    تحسب الانحراف المالي لكل حالة DRG:
    drg_variance = (claims_amount - expected_cost) / expected_cost * 100
    تحاول استخدام expected_cost أو drg_expected_cost إن وجد.
    """
    df = df.copy()
    status = "ok"

    if "claims_amount" not in df.columns:
        df["drg_variance"] = np.nan
        status = "no_claims"
        return df, status

    # نحاول استخدام expected_cost أو drg_expected_cost كـ benchmark
    if "expected_cost" in df.columns:
        base = pd.to_numeric(df["expected_cost"], errors="coerce")
    elif "drg_expected_cost" in df.columns:
        base = pd.to_numeric(df["drg_expected_cost"], errors="coerce")
    else:
        df["drg_variance"] = np.nan
        status = "no_expected"
        return df, status

    claims = pd.to_numeric(df["claims_amount"], errors="coerce")
    df["drg_variance"] = (claims - base) / (base + 1e-9) * 100.0
    return df, status


def compute_cars(df: pd.DataFrame) -> pd.DataFrame:
    """
    تبني CARS من:
    - CRS (خطر سريري)
    - PCS (توثيق/التزام مقدم الخدمة)
    - drg_variance (انحراف مالي على مستوى DRG)

    الصيغة الحالية تجريبية ويمكن تعديل الأوزان لاحقًا.
    """
    df = df.copy()

    if "drg_variance" in df.columns and df["drg_variance"].notna().any():
        # نحدّ drg_variance في مدى [-200 , 200] ثم نطبّعها إلى 0–100 داخل العينة
        dv = df["drg_variance"].clip(-200, 200)
        dv_norm = 100 * (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)

        df["CARS"] = (
            0.5 * pd.to_numeric(df["CRS"], errors="coerce") +
            0.3 * pd.to_numeric(df["PCS"], errors="coerce") +
            0.2 * dv_norm
        )
    else:
        # لا توجد معلومات DRG variance:
        # إن كان CARS موجوداً من المصدر نستخدمه، وإلا نساويه بـ CRS
        if "CARS" in df.columns:
            df["CARS"] = pd.to_numeric(df["CARS"], errors="coerce")
        else:
            df["CARS"] = pd.to_numeric(df["CRS"], errors="coerce")

    return df


st.sidebar.markdown("### UPI Weights")
wBAS = st.sidebar.slider("BAS weight", 0.0, 1.0, 0.25)
wCRS = st.sidebar.slider("CRS weight", 0.0, 1.0, 0.25)
wCARS = st.sidebar.slider("CARS weight", 0.0, 1.0, 0.25)
wPEN = st.sidebar.slider("Provider Penalty weight", 0.0, 1.0, 0.15)
wFEI = st.sidebar.slider("FEI weight", 0.0, 1.0, 0.10)

total_w = wBAS + wCRS + wCARS + wPEN + wFEI
if abs(total_w - 1.0) > 0.02:
    st.sidebar.warning(f"Total weights = {total_w:.2f}. Ideal = 1.00")

st.sidebar.markdown("### Risk thresholds")
high_thr = st.sidebar.slider("High-risk threshold", 50, 95, 80)
med_thr = st.sidebar.slider("Medium-risk threshold", 40, 89, 60)


# =====================================================================
# PIPELINE: DRG variance → CARS → Normalization → UPI
# =====================================================================

# 1) حساب DRG variance إن أمكن
data, drg_status = compute_drg_variance(data)

if drg_status == "no_claims":
    st.info("Column 'claims_amount' not found – DRG variance will be NaN.")
elif drg_status == "no_expected":
    st.info("Neither 'expected_cost' nor 'drg_expected_cost' found – DRG variance will be NaN.")

# 2) تطبيع BAS/CRS/PCS/PPS/FEI
for col in ["BAS", "CRS", "PCS", "PPS", "FEI"]:
    data[col] = normalize(data[col])

# 3) حساب CARS من CRS/PCS + drg_variance
data = compute_cars(data)

# 4) تطبيع CARS كذلك
data["CARS"] = normalize(data["CARS"])


# =====================================================================
# CALCULATIONS: Provider penalty + UPI + Risk Level
# =====================================================================
def calc_provider_penalty(r: pd.Series) -> float:
    return 0.6 * (100 - r["PCS"]) + 0.4 * (100 - r["PPS"])

def calc_upi(r: pd.Series) -> float:
    return (
        wBAS * r["BAS"]
        + wCRS * r["CRS"]
        + wCARS * r["CARS"]
        + wPEN * r["provider_penalty"]
        + wFEI * r["FEI"]
    )

def classify_risk(u: float) -> str:
    if u >= high_thr:
        return "High Risk"
    if u >= med_thr:
        return "Medium Risk"
    return "Low Risk"


data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)
data["UPI"] = data.apply(calc_upi, axis=1)
data["risk_level"] = data["UPI"].apply(classify_risk)


# =====================================================================
# LOGGING SYSTEM
# =====================================================================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "upi_runs_log.csv")

def log_run(df: pd.DataFrame, dataset_name: str):
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
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


# =====================================================================
# TABS (5 Tabs)
# =====================================================================
tab1, tab2, tab3, tab_drg, tab_logs = st.tabs([
    "Level 1 – Strategic Executive Dashboard",
    "Level 2 – Facility Performance",
    "Level 3 – Patient Risk Panel",
    "DRG-Level Risk Comparison",
    "Model Governance / Logs",
])


# ---------------------------------------------------------------------
# LEVEL 1 – STRATEGIC
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Level 1 – Strategic Executive Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", len(data))
    c2.metric("High-Risk", int((data["risk_level"] == "High Risk").sum()))
    c3.metric("Average UPI", f"{data['UPI'].mean():.1f}")

    if st.button("Log this run (append to logs/upi_runs_log.csv)"):
        log_run(data, dataset_name)
        st.success("Run logged successfully")

    st.markdown("---")

    st.markdown("### 1) UPI Trend")
    if has_period:
        trend = data.groupby("period")["UPI"].mean().reset_index()
        st.plotly_chart(px.line(trend, x="period", y="UPI", markers=True),
                        use_container_width=True)
    else:
        st.info("Add 'period' column to enable UPI Trend")

    st.markdown("### 2) Loss Ratio vs UPI")
    if has_claims and has_premium and has_period:
        grp = data.groupby("period").agg(
            avg_upi=("UPI", "mean"),
            claims=("claims_amount", "sum"),
            prem=("premium_amount", "sum"),
        ).reset_index()
        grp["loss_ratio"] = 100 * grp["claims"] / grp["prem"].replace(0, np.nan)
        st.plotly_chart(px.line(grp, x="period", y=["avg_upi", "loss_ratio"]),
                        use_container_width=True)
    else:
        st.info("Need claims_amount + premium_amount + period to show Loss Ratio vs UPI.")

    st.markdown("### 3) Average UPI by Region")
    if has_region:
        rr = data.groupby("region")["UPI"].mean().reset_index()
        st.plotly_chart(px.bar(rr, x="region", y="UPI"),
                        use_container_width=True)
    else:
        st.info("Add 'region' column to enable regional view.")

    st.markdown("### 4) Top 10 Providers by UPI")
    if has_provider:
        prov = (
            data.groupby("provider_name")["UPI"]
            .mean()
            .reset_index()
            .sort_values("UPI", ascending=False)
            .head(10)
        )
        st.plotly_chart(px.bar(prov, x="provider_name", y="UPI"),
                        use_container_width=True)
        st.dataframe(prov, use_container_width=True)
    else:
        st.info("Add 'provider_name' column to enable provider ranking.")


# ---------------------------------------------------------------------
# LEVEL 2 – FACILITY PERFORMANCE
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Level 2 – Facility Performance")

    if not has_provider:
        st.info("Add 'provider_name' column to enable facility analytics.")
    else:
        providers = sorted(data["provider_name"].unique())
        sel_provider = st.selectbox("Select Provider", providers)

        d = data[data["provider_name"] == sel_provider]

        c1, c2, c3 = st.columns(3)
        c1.metric("Patients", len(d))
        c2.metric("High-Risk", int((d["risk_level"] == "High Risk").sum()))
        c3.metric("Avg UPI", f"{d['UPI'].mean():.1f}")

        st.markdown("---")

        comp = d[["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]].mean().reset_index()
        comp.columns = ["Score", "Value"]
        st.markdown("### Risk Composition (BAS / CRS / CARS / PCS / PPS / FEI)")
        st.plotly_chart(px.bar(comp, x="Score", y="Value"), use_container_width=True)

        st.markdown("### Provider vs Network UPI")
        colA, colB = st.columns(2)
        colA.metric("Provider Avg UPI", f"{d['UPI'].mean():.1f}")
        colB.metric("Network Avg UPI", f"{data['UPI'].mean():.1f}")

        st.markdown("### DRG Shift (this provider)")
        if has_drg and has_period:
            drg = d.groupby(["period", "drg_group"])["UPI"].mean().reset_index()
            st.plotly_chart(px.line(drg, x="period", y="UPI", color="drg_group", markers=True),
                            use_container_width=True)
        else:
            st.info("Need 'drg_group' + 'period' columns for DRG shift.")

        st.markdown("### PCS Trend (documentation/compliance)")
        if has_period:
            pcs = d.groupby("period")["PCS"].mean().reset_index()
            st.plotly_chart(px.line(pcs, x="period", y="PCS", markers=True),
                            use_container_width=True)
        else:
            st.info("Need 'period' column to show PCS trend.")


# ---------------------------------------------------------------------
# LEVEL 3 – PATIENT PANEL
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Level 3 – Patient Risk Panel")

    patients = sorted(data["patient_id"].unique())
    sel_pt = st.selectbox("Select Patient", patients)

    pt = data[data["patient_id"] == sel_pt]
    if has_period:
        pt = pt.sort_values("period")

    latest = pt.iloc[-1]

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
    st.markdown("### Risk Drivers (latest record)")
    st.plotly_chart(px.bar(drivers, x="Score", y="Value"),
                    use_container_width=True)

    st.markdown("### Provider Penalty")
    c4, c5 = st.columns(2)
    c4.metric("PCS", f"{latest['PCS']:.1f}")
    c5.metric("Penalty", f"{latest['provider_penalty']:.1f}")

    st.markdown("### Recent 3 Periods (UPI Sparkline)")
    if has_period and len(pt) >= 3:
        last3 = pt.tail(3)
        st.plotly_chart(px.line(last3, x="period", y="UPI", markers=True),
                        use_container_width=True)
    else:
        st.info("Not enough period history for this patient.")


# ---------------------------------------------------------------------
# DRG-LEVEL RISK COMPARISON
# ---------------------------------------------------------------------
with tab_drg:
    st.subheader("DRG-Level Risk Comparison (Network & Providers)")

    if not has_drg:
        st.info("No 'drg_group' column found. Add DRG grouping to enable this tab.")
    else:
        # 1) Network-level DRG ranking by UPI
        st.markdown("### 1) Network-level DRG risk ranking (by UPI)")
        drg_rank = (
            data.groupby("drg_group")["UPI"]
            .mean()
            .reset_index()
            .sort_values("UPI", ascending=False)
        )
        st.plotly_chart(
            px.bar(drg_rank, x="drg_group", y="UPI",
                   title="Average UPI by DRG (Network)",
                   labels={"drg_group": "DRG Group", "UPI": "Average UPI"}),
            use_container_width=True,
        )
        st.dataframe(drg_rank, use_container_width=True)

        st.markdown("---")

        # 2) Network-level DRG ranking by DRG variance (financial)
        if "drg_variance" in data.columns and data["drg_variance"].notna().any():
            st.markdown("### 2) DRG financial variance ranking (Network)")
            drg_var_rank = (
                data.groupby("drg_group")["drg_variance"]
                .mean()
                .reset_index()
                .sort_values("drg_variance", ascending=False)
            )
            st.plotly_chart(
                px.bar(
                    drg_var_rank,
                    x="drg_group",
                    y="drg_variance",
                    title="Average DRG Variance by DRG (Network)",
                    labels={"drg_group": "DRG Group", "drg_variance": "Avg DRG Variance (%)"},
                ),
                use_container_width=True,
            )
            st.dataframe(drg_var_rank, use_container_width=True)

        st.markdown("---")

        # 3) Provider × DRG – UPI Heatmap
        st.markdown("### 3) Provider × DRG – Heatmap of Average UPI")
        if has_provider:
            matrix = (
                data.groupby(["provider_name", "drg_group"])["UPI"]
                .mean()
                .reset_index()
            )
            pivot = matrix.pivot(index="provider_name", columns="drg_group", values="UPI")

            st.write("Average UPI per Provider/DRG (table):")
            st.dataframe(pivot, use_container_width=True)

            fig_hm = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(x="DRG Group", y="Provider", color="Avg UPI"),
                title="Provider × DRG – UPI Heatmap (higher = more risk)",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Add 'provider_name' column to enable Provider × DRG matrix.")

        st.markdown("---")

        # 4) DRG risk profile for selected provider
        if has_provider:
            st.markdown("### 4) DRG Risk Profile for Selected Provider")
            providers = sorted(data["provider_name"].unique())
            sel_prov_drg = st.selectbox("Select provider for DRG profile", providers, key="prov_drg_profile")

            df_prov = data[data["provider_name"] == sel_prov_drg]
            drg_prov = (
                df_prov.groupby("drg_group")["UPI"]
                .mean()
                .reset_index()
                .sort_values("UPI", ascending=False)
            )
            st.plotly_chart(
                px.bar(
                    drg_prov,
                    x="drg_group",
                    y="UPI",
                    title=f"Average UPI by DRG – {sel_prov_drg}",
                    labels={"drg_group": "DRG Group", "UPI": "Average UPI"},
                ),
                use_container_width=True,
            )
            st.dataframe(drg_prov, use_container_width=True)
        st.markdown("---")

        # 5) DRG risk trend across time (network-level)
        st.markdown("### 5) DRG Risk Trend over Time (Network)")
        if has_period:
            drg_time = (
                data.groupby(["period", "drg_group"])["UPI"]
                .mean()
                .reset_index()
                .sort_values(["period", "drg_group"])
            )
            fig_dt = px.line(
                drg_time,
                x="period",
                y="UPI",
                color="drg_group",
                markers=True,
                title="Average UPI by DRG over time (Network)",
                labels={"period": "Period", "UPI": "Avg UPI", "drg_group": "DRG"},
            )
            st.plotly_chart(fig_dt, use_container_width=True)
            st.dataframe(drg_time, use_container_width=True)
        else:
            st.info("Add 'period' column to enable DRG risk trend over time.")


# ---------------------------------------------------------------------
# MODEL GOVERNANCE / LOGS
# ---------------------------------------------------------------------
with tab_logs:
    st.subheader("Model Governance – Run Logs")

    if not os.path.isfile(LOG_FILE):
        st.info("No logs found. Click 'Log this run' in Level 1.")
    else:
        logs = pd.read_csv(LOG_FILE)

        if "timestamp" in logs.columns:
            logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")

        logs = logs.sort_values("timestamp")

        colA, colB = st.columns(2)

        # Download logs
        buf = io.StringIO()
        logs.to_csv(buf, index=False)
        colA.download_button(
            "Download full logs (CSV)",
            buf.getvalue(),
            "upi_runs_log.csv",
            "text/csv",
        )

        # Clear logs (Admin only)
        if IS_ADMIN:
            if colB.button("Clear logs (Admin only)"):
                os.remove(LOG_FILE)
                st.warning("Logs cleared.")
                st.stop()
        else:
            colB.caption("Admin only to clear logs.")

        st.markdown("---")

        total_runs = len(logs)
        last_time = logs["timestamp"].max()
        last_avg = logs["avg_upi"].iloc[-1] if "avg_upi" in logs.columns else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Runs", total_runs)
        c2.metric("Last Run", str(last_time))
        c3.metric("Last Avg UPI", f"{last_avg:.2f}" if last_avg is not None else "N/A")

        st.markdown("---")

        if {"timestamp", "avg_upi"}.issubset(logs.columns):
            st.markdown("### Avg UPI across runs")
            st.plotly_chart(px.line(logs, x="timestamp", y="avg_upi", markers=True),
                            use_container_width=True)

        if {"timestamp", "high_risk_pct"}.issubset(logs.columns):
            st.markdown("### High-Risk % across runs")
            st.plotly_chart(px.line(logs, x="timestamp", y="high_risk_pct", markers=True),
                            use_container_width=True)

        if "norm_method" in logs.columns:
            st.markdown("### Normalization methods used")
            st.plotly_chart(px.histogram(logs, x="norm_method"),
                            use_container_width=True)

        if "model_version" in logs.columns:
            st.markdown("### Model versions")
            st.plotly_chart(px.histogram(logs, x="model_version"),
                            use_container_width=True)

        st.markdown("### Latest 20 Runs")
        st.dataframe(
            logs.sort_values("timestamp", ascending=False).head(20),
            use_container_width=True,
        )


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>MHAAMB</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model " + MODEL_VERSION + " • " + dt.datetime.now().strftime("%Y-%m-%d"))
