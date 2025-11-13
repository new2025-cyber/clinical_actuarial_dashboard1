import io
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_autorefresh import st_autorefresh


# =========================================================
# ADMIN KEY (اختياري – آمن محلياً)
# =========================================================
def get_admin_key():
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        return ""  # لا يوجد secrets محلياً


ADMIN_KEY = get_admin_key()

st.set_page_config(page_title="Clinical–Actuarial UPI Dashboard", layout="wide")
st.title("Clinical–Actuarial UPI Dashboard – 3-Level View")

admin_input = st.sidebar.text_input("Admin key (optional)", type="password")
IS_ADMIN = ADMIN_KEY != "" and admin_input == ADMIN_KEY

if not IS_ADMIN:
    st_autorefresh(interval=15 * 60 * 1000, key="public_refresh")
    st.sidebar.caption("Public view (auto-refresh every 15 minutes).")
else:
    st.sidebar.caption("Admin mode active.")


# =========================================================
# ملف البيانات
# =========================================================
uploaded = st.file_uploader("Upload Patient File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Please upload a CSV/Excel file containing patient scores.")
    st.stop()

ext = uploaded.name.split(".")[-1].lower()
if ext == "csv":
    data = pd.read_csv(uploaded)
else:
    data = pd.read_excel(uploaded)

# الأعمدة الأساسية المطلوبة لحساب UPI
required_cols = ["patient_id", "BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# أعمدة اختيارية للمستويات الأعلى
has_region = "region" in data.columns          # لمستوى 1 (Regional risk)
has_provider_name = "provider_name" in data.columns  # لمستوى 1+2
has_period = "period" in data.columns          # لكل مستويات الـ trend
has_drg = "drg_group" in data.columns          # لمستوى 2 (DRG shift)
has_claims = "claims_amount" in data.columns
has_premium = "premium_amount" in data.columns


# =========================================================
# Normalize & Weights
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
wBAS = st.sidebar.slider("Weight BAS", 0.0, 1.0, 0.25, 0.01)
wCRS = st.sidebar.slider("Weight CRS", 0.0, 1.0, 0.25, 0.01)
wCARS = st.sidebar.slider("Weight CARS", 0.0, 1.0, 0.25, 0.01)
wPEN = st.sidebar.slider("Weight Provider Penalty", 0.0, 1.0, 0.15, 0.01)
wFEI = st.sidebar.slider("Weight FEI", 0.0, 1.0, 0.10, 0.01)

total_w = wBAS + wCRS + wCARS + wPEN + wFEI
if abs(total_w - 1.0) > 0.02:
    st.sidebar.warning(f"Total weights = {total_w:.2f}. Ideally sum ≈ 1.00")


st.sidebar.markdown("### Risk thresholds")
high_thr = st.sidebar.slider("High risk threshold", 50, 95, 80, 1)
med_thr = st.sidebar.slider("Medium risk threshold", 40, 89, 60, 1)


# =========================================================
# Provider Penalty + UPI
# =========================================================
def calc_provider_penalty(row) -> float:
    return 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS"])


def calc_upi(row) -> float:
    return (
        wBAS * row["BAS"]
        + wCRS * row["CRS"]
        + wCARS * row["CARS"]
        + wPEN * row["provider_penalty"]
        + wFEI * row["FEI"]
    )


def classify_risk(upi: float) -> str:
    if upi >= high_thr:
        return "High Risk"
    if upi >= med_thr:
        return "Medium Risk"
    return "Low Risk"


data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)
data["UPI"] = data.apply(calc_upi, axis=1)
data["risk_level"] = data["UPI"].apply(classify_risk)


# =========================================================
# Tabs: 3 مستويات
# =========================================================
tab_strat, tab_fac, tab_patient = st.tabs(
    [
        "Level 1 – Strategic Executive Dashboard",
        "Level 2 – Facility Performance",
        "Level 3 – Patient Risk Panel",
    ]
)

# ---------------------------------------------------------
# LEVEL 1 – STRATEGIC EXECUTIVE DASHBOARD
# ---------------------------------------------------------
with tab_strat:
    st.subheader("Level 1 – Strategic Executive Dashboard")

    # --- KPIs ---
    c1, c2, c3 = st.columns(3)
    total_patients = len(data)
    high_risk_patients = int((data["risk_level"] == "High Risk").sum())
    avg_upi = float(data["UPI"].mean())

    c1.metric("Total Patients", total_patients)
    c2.metric("High-Risk Patients", high_risk_patients)
    c3.metric("Average UPI", f"{avg_upi:.1f}")

    st.markdown("---")

    # 1) UPI Trend (central index)
    st.markdown("### 1) Unified Predictive Index – UPI Trend")
    if has_period:
        trend = (
            data.groupby("period")["UPI"]
            .mean()
            .reset_index()
            .sort_values("period")
        )
        fig_trend = px.line(
            trend,
            x="period",
            y="UPI",
            markers=True,
            title="UPI Trend over time",
            template="plotly_white",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No 'period' column found. Add a period/month field to enable UPI trend.")

    # 2) Loss Ratio vs UPI
    st.markdown("### 2) Loss Ratio vs UPI (Financial Impact)")
    if has_claims and has_premium and has_period:
        agg_fin = (
            data.groupby("period")
            .agg(
                avg_upi=("UPI", "mean"),
                claims=("claims_amount", "sum"),
                premium=("premium_amount", "sum"),
            )
            .reset_index()
            .sort_values("period")
        )
        agg_fin["loss_ratio"] = 100 * agg_fin["claims"] / agg_fin["premium"].replace(0, np.nan)

        fig_lr = px.line(
            agg_fin,
            x="period",
            y=["avg_upi", "loss_ratio"],
            title="Loss Ratio vs UPI",
            labels={"value": "Value", "variable": "Metric"},
            template="plotly_white",
        )
        st.plotly_chart(fig_lr, use_container_width=True)
    else:
        st.info("To enable Loss Ratio vs UPI, add 'claims_amount', 'premium_amount', and 'period' columns.")

    # 3) Regional Risk Heatmap (simplified as bar chart)
    st.markdown("### 3) Regional Risk Overview")
    if has_region:
        reg = data.groupby("region")["UPI"].mean().reset_index()
        fig_reg = px.bar(
            reg,
            x="region",
            y="UPI",
            title="Average UPI by Region",
            template="plotly_white",
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("No 'region' column found. Add region/province to enable regional risk view.")

    # 4) Provider Risk Ranking (Top 10 Hospitals)
    st.markdown("### 4) Provider Risk Ranking – Top 10 Hospitals")
    if has_provider_name:
        prov = (
            data.groupby("provider_name")["UPI"]
            .mean()
            .reset_index()
            .sort_values("UPI", ascending=False)
            .head(10)
        )
        fig_top = px.bar(
            prov,
            x="provider_name",
            y="UPI",
            title="Top 10 Providers by Average UPI",
            template="plotly_white",
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.dataframe(prov, use_container_width=True)
    else:
        st.info("No 'provider_name' column found. Add it to enable provider ranking.")

    st.markdown("---")

    # Download section (CSV + PDF)
    st.markdown("### Downloads (Network-level Summary)")

    # CSV
    csv_buf = io.StringIO()
    data.to_csv(csv_buf, index=False)
    st.download_button(
        "Download full dataset (CSV)",
        data=csv_buf.getvalue(),
        file_name="upi_full_dataset.csv",
        mime="text/csv",
    )

    # PDF Summary
    def generate_pdf_summary(df: pd.DataFrame) -> io.BytesIO:
        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=letter)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(30, 750, "Clinical–Actuarial UPI – Strategic Summary")

        pdf.setFont("Helvetica", 11)
        pdf.drawString(30, 730, f"Total Patients: {len(df)}")
        pdf.drawString(30, 715, f"High-Risk Patients: {(df['risk_level']=='High Risk').sum()}")
        pdf.drawString(30, 700, f"Average UPI: {df['UPI'].mean():.2f}")
        pdf.drawString(30, 680, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

        pdf.drawString(30, 655, f"High threshold: {high_thr}   Medium threshold: {med_thr}")
        pdf.drawString(30, 640, f"Normalization: {norm_method}")
        pdf.drawString(30, 625, f"Weights – BAS:{wBAS:.2f}, CRS:{wCRS:.2f}, CARS:{wCARS:.2f}, PEN:{wPEN:.2f}, FEI:{wFEI:.2f}")

        pdf.showPage()
        pdf.save()
        buf.seek(0)
        return buf

    if st.button("Generate Strategic PDF Summary"):
        pdf_file = generate_pdf_summary(data)
        st.download_button(
            "Download Strategic PDF Summary",
            data=pdf_file,
            file_name="upi_strategic_summary.pdf",
            mime="application/pdf",
        )


# ---------------------------------------------------------
# LEVEL 2 – FACILITY PERFORMANCE DASHBOARD
# ---------------------------------------------------------
with tab_fac:
    st.subheader("Level 2 – Facility Performance Dashboard")

    if not has_provider_name:
        st.info("No 'provider_name' column found in the dataset. Add it to enable facility-level analytics.")
    else:
        providers = sorted(data["provider_name"].dropna().unique())
        selected_provider = st.selectbox("Select provider / hospital", providers)

        df_p = data[data["provider_name"] == selected_provider].copy()

        if df_p.empty:
            st.warning("No patients found for this provider.")
        else:
            # KPIs for provider
            c1, c2, c3 = st.columns(3)
            c1.metric("Patients (provider)", len(df_p))
            c2.metric("High-Risk (provider)", int((df_p["risk_level"] == "High Risk").sum()))
            c3.metric("Average UPI (provider)", f"{df_p['UPI'].mean():.1f}")

            st.markdown("---")

            # 1) Risk Composition (BAS+CRS+... for this provider)
            st.markdown("### 1) Risk Composition (scores for this provider)")
            comp = (
                df_p[["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]]
                .mean()
                .reset_index()
            )
            comp.columns = ["Score", "Value"]
            fig_comp = px.bar(
                comp,
                x="Score",
                y="Value",
                title="Average Score Composition – Current Provider",
                template="plotly_white",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # 2) UPI vs Network Average (simple comparison)
            st.markdown("### 2) UPI vs Network Average")
            net_avg_upi = data["UPI"].mean()
            prov_avg_upi = df_p["UPI"].mean()
            c4, c5 = st.columns(2)
            c4.metric("Provider Avg UPI", f"{prov_avg_upi:.1f}")
            c5.metric("Network Avg UPI", f"{net_avg_upi:.1f}")

            # 3) DRG Risk Shift (Monthly) – if drg_group+period exist
            st.markdown("### 3) DRG Risk Shift (Monthly)")

            if has_drg and has_period:
                drg_shift = (
                    df_p.groupby(["period", "drg_group"])["UPI"]
                    .mean()
                    .reset_index()
                    .sort_values(["period", "drg_group"])
                )
                fig_drg = px.line(
                    drg_shift,
                    x="period",
                    y="UPI",
                    color="drg_group",
                    markers=True,
                    title="Average UPI by DRG over time (this provider)",
                    template="plotly_white",
                )
                st.plotly_chart(fig_drg, use_container_width=True)
                st.dataframe(drg_shift, use_container_width=True)
            else:
                st.info("To enable DRG Risk Shift, add 'drg_group' and 'period' columns.")

            # 4) Provider Compliance Score – PCS Trend
            st.markdown("### 4) Provider Compliance Score – PCS Trend")
            if has_period:
                pcs_trend = (
                    df_p.groupby("period")["PCS"]
                    .mean()
                    .reset_index()
                    .sort_values("period")
                )
                fig_pcs = px.line(
                    pcs_trend,
                    x="period",
                    y="PCS",
                    markers=True,
                    title="PCS Trend for this provider (documentation/compliance)",
                    template="plotly_white",
                )
                st.plotly_chart(fig_pcs, use_container_width=True)
                st.dataframe(pcs_trend, use_container_width=True)
            else:
                st.info("No 'period' column found – cannot show PCS trend.")


# ---------------------------------------------------------
# LEVEL 3 – PATIENT RISK PANEL
# ---------------------------------------------------------
with tab_patient:
    st.subheader("Level 3 – Patient Risk Panel")

    patients = sorted(data["patient_id"].dropna().unique())
    selected_patient = st.selectbox("Select patient", patients)

    if has_period:
        df_pt = data[data["patient_id"] == selected_patient].copy().sort_values("period")
    else:
        df_pt = data[data["patient_id"] == selected_patient].copy()

    if df_pt.empty:
        st.warning("No records found for this patient.")
    else:
        latest = df_pt.iloc[-1]

        # UPI gauge (رقم + مستوى)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current UPI", f"{latest['UPI']:.1f}")
        c2.metric("Risk Level", latest["risk_level"])
        if has_provider_name:
            c3.metric("Current Provider", str(latest["provider_name"]))
        else:
            c3.metric("Current Provider", "N/A")

        st.markdown("---")

        # 2) Risk Drivers (BAS/CRS/CARS/PCS/PPS/FEI)
        st.markdown("### 2) Risk Drivers (latest record)")

        drivers = pd.DataFrame(
            {
                "Score": ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"],
                "Value": [
                    latest["BAS"],
                    latest["CRS"],
                    latest["CARS"],
                    latest["PCS"],
                    latest["PPS"],
                    latest["FEI"],
                ],
            }
        )
        fig_drv = px.bar(
            drivers,
            x="Score",
            y="Value",
            title="Risk Components for this patient",
            template="plotly_white",
        )
        st.plotly_chart(fig_drv, use_container_width=True)
        st.dataframe(drivers, use_container_width=True)

        # 3) Influence of Provider (PCS / provider_penalty)
        st.markdown("### 3) Influence of Provider (PCS & Penalty)")
        c4, c5 = st.columns(2)
        c4.metric("PCS (documentation/compliance)", f"{latest['PCS']:.1f}")
        c5.metric("Provider penalty", f"{latest['provider_penalty']:.1f}")

        # 4) Early Risk Movement (last 3 periods)
        st.markdown("### 4) Early Risk Movement (last 3 periods)")
        if has_period and len(df_pt) > 1:
            last3 = df_pt.tail(3)
            fig_sp = px.line(
                last3,
                x="period",
                y="UPI",
                markers=True,
                title="UPI – last 3 periods",
                template="plotly_white",
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.dataframe(last3[["period", "UPI", "risk_level"]], use_container_width=True)
        else:
            st.info("Not enough period data to show last-3-period movement.")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>Mudather</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model v2.0 • " + dt.datetime.now().strftime("%Y-%m-%d"))

