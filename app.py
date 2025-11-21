import os
import csv
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================================
# ADMIN & VERSION
# =====================================================================


def get_admin_key() -> str:
    """Reads ADMIN_KEY from Streamlit secrets if available."""
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        return ""


ADMIN_KEY = get_admin_key()
MODEL_VERSION = "v3.0-MVP-DRG-UI"

st.set_page_config(
    page_title="Clinical–Actuarial Risk Intelligence Dashboard – UPI & DRG Variance",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM UI STYLE (COLORS & CARDS – Bupa-inspired)
# =====================================================================

PRIMARY_COLOR = "#0096D6"   # main blue
PRIMARY_DARK = "#005B9A"    # darker blue
ACCENT_COLOR = "#5BC2E7"    # light cyan
BG_COLOR = "#f4f9fc"        # very light blue background

custom_css = f"""
<style>

body {{
    background-color: {BG_COLOR};
    font-family: "Cairo", "Segoe UI", sans-serif;
}}

.top-bar {{
    background: linear-gradient(90deg, {PRIMARY_COLOR}, {PRIMARY_DARK});
    padding: 18px 26px;
    border-radius: 0 0 18px 18px;
    color: #ffffff;
    margin-bottom: 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.18);
}}
.top-bar-title {{
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 2px;
}}
.top-bar-sub {{
    font-size: 13px;
    opacity: 0.92;
}}
.top-bar-pill {{
    float: right;
    background-color: rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 12px;
    margin-top: -24px;
}}

h2, h3, h4 {{
    color: {PRIMARY_DARK};
}}

.card {{
    background-color: #ffffff;
    padding: 18px 20px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    text-align: center;
    margin-bottom: 16px;
    border-top: 4px solid {PRIMARY_COLOR};
}}

.card h2 {{
    font-size: 30px;
    color: {PRIMARY_COLOR};
    margin-bottom: 4px;
    margin-top: 0px;
}}

.card p {{
    color: #555555;
    margin-top: 0;
    font-size: 14px;
}}

.card-accent {{
    border-top: 4px solid {ACCENT_COLOR};
}}

.card-small-number {{
    font-size: 18px;
    color: {ACCENT_COLOR};
    font-weight: 600;
}}

div.stTabs [data-baseweb="tab"] {{
    font-size: 15px;
    padding: 8px 14px;
}}

.footer {{
    text-align: center;
    color: #777777;
    padding-top: 24px;
    padding-bottom: 8px;
    font-size: 13px;
}}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

top_bar_html = f"""
<div class="top-bar">
  <div class="top-bar-title">Clinical–Actuarial Risk Intelligence Dashboard</div>
  <div class="top-bar-sub">
    UPI &amp; DRG Variance • Unified Patient &amp; Provider Risk • MVP – Internal Demo
  </div>
  <div class="top-bar-pill">
    Experimental • {MODEL_VERSION}
  </div>
</div>
"""
st.markdown(top_bar_html, unsafe_allow_html=True)

st.info(
    "⚠️ This is an internal MVP prototype based on mock or preliminary data – "
    "intended for concept demonstration, not for final decision-making."
)

# =====================================================================
# ADMIN INPUT
# =====================================================================

admin_input = st.sidebar.text_input("Admin key (optional)", type="password")
IS_ADMIN = ADMIN_KEY != "" and admin_input == ADMIN_KEY

if IS_ADMIN:
    st.sidebar.success("Admin mode active")
else:
    st.sidebar.caption("Public mode – admin features disabled")

# =====================================================================
# DATA & MODEL CONFIGURATION
# =====================================================================

st.markdown("### Data & Model Configuration")
st.markdown("#### 1) Upload Patient-Level Dataset")

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Please upload a dataset file to continue.")
    st.stop()

dataset_name = uploaded.name
ext = dataset_name.split(".")[-1].lower()

if ext == "csv":
    data = pd.read_csv(uploaded)
else:
    data = pd.read_excel(uploaded)

required_base = [
    "patient_id",
    "BAS",
    "CRS",
    "PCS",
    "PPS",
    "FEI",
    "claims_amount",
    "provider_name",
    "drg_group",
    "period",
]

missing_base = [c for c in required_base if c not in data.columns]
if missing_base:
    st.error(f"Missing required columns: {missing_base}")
    st.stop()

if "expected_cost" in data.columns:
    expected_col = "expected_cost"
elif "drg_expected_cost" in data.columns:
    expected_col = "drg_expected_cost"
else:
    st.error("Missing expected cost column: please add 'expected_cost' or 'drg_expected_cost'.")
    st.stop()

has_region = "region" in data.columns
has_premium = "premium_amount" in data.columns

# =====================================================================
# NORMALIZATION & WEIGHTS
# =====================================================================

st.sidebar.markdown("### Normalization method")

norm_method = st.sidebar.selectbox(
    "Choose normalization",
    ["MinMax (0–100)", "Z-score (50±10)", "None"],
    index=0,
)


def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if norm_method == "None":
        return s
    if norm_method == "MinMax (0–100)":
        return 100.0 * (s - s.min()) / (s.max() - s.min() + 1e-9)
    if norm_method == "Z-score (50±10)":
        z = (s - s.mean()) / (s.std() + 1e-9)
        return 50.0 + 10.0 * z
    return s


st.sidebar.markdown("### UPI Weights")

wBAS = st.sidebar.slider("Weight BAS", 0.0, 1.0, 0.25)
wCRS = st.sidebar.slider("Weight CRS", 0.0, 1.0, 0.25)
wCARS = st.sidebar.slider("Weight CARS", 0.0, 1.0, 0.25)
wPEN = st.sidebar.slider("Weight Provider Penalty", 0.0, 1.0, 0.15)
wFEI = st.sidebar.slider("Weight FEI", 0.0, 1.0, 0.10)

total_w = wBAS + wCRS + wCARS + wPEN + wFEI
if abs(total_w - 1.0) > 0.02:
    st.sidebar.warning(f"Total weights = {total_w:.2f} (ideal = 1.00).")

st.sidebar.markdown("### Risk thresholds")

high_thr = st.sidebar.slider("High risk ≥", 50, 95, 80)
med_thr = st.sidebar.slider("Medium risk ≥", 40, 89, 60)

# =====================================================================
# DRG VARIANCE + CARS
# =====================================================================


def compute_drg_variance(df: pd.DataFrame, claims_col: str, exp_col: str) -> pd.DataFrame:
    out = df.copy()
    claims = pd.to_numeric(out[claims_col], errors="coerce")
    expected = pd.to_numeric(out[exp_col], errors="coerce")
    out["drg_variance"] = (claims - expected) / (expected + 1e-9) * 100.0
    return out


def compute_cars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build CARS from:
    - CRS (clinical risk)
    - PCS (provider compliance)
    - drg_variance (financial deviation)
    """
    out = df.copy()
    crs = pd.to_numeric(out["CRS"], errors="coerce")
    pcs = pd.to_numeric(out["PCS"], errors="coerce")

    if "drg_variance" in out.columns and out["drg_variance"].notna().any():
        dv = pd.to_numeric(out["drg_variance"], errors="coerce").clip(-200, 200)
        dv_norm = 100.0 * (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
        out["CARS"] = 0.5 * crs + 0.3 * pcs + 0.2 * dv_norm
    else:
        out["CARS"] = crs

    return out


for col in ["BAS", "CRS", "PCS", "PPS", "FEI"]:
    data[col] = normalize(data[col])

data = compute_drg_variance(data, "claims_amount", expected_col)
data = compute_cars(data)
data["CARS"] = normalize(data["CARS"])

# =====================================================================
# PROVIDER PENALTY + UPI + RISK CLASS
# =====================================================================


def calc_provider_penalty(row: pd.Series) -> float:
    return 0.6 * (100.0 - row["PCS"]) + 0.4 * (100.0 - row["PPS"])


def calc_upi(row: pd.Series) -> float:
    return (
        wBAS * row["BAS"]
        + wCRS * row["CRS"]
        + wCARS * row["CARS"]
        + wPEN * row["provider_penalty"]
        + wFEI * row["FEI"]
    )


def classify_risk(upi: float) -> str:
    if upi >= float(high_thr):
        return "High Risk"
    if upi >= float(med_thr):
        return "Medium Risk"
    return "Low Risk"


data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)
data["UPI"] = data.apply(calc_upi, axis=1)
data["risk_level"] = data["UPI"].apply(classify_risk)

total_patients = len(data)
num_high = int((data["risk_level"] == "High Risk").sum())
num_med = int((data["risk_level"] == "Medium Risk").sum())
num_low = int((data["risk_level"] == "Low Risk").sum())
avg_upi = float(data["UPI"].mean()) if total_patients > 0 else 0.0

# =====================================================================
# LOGGING
# =====================================================================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "upi_runs_log.csv")


def log_run(df: pd.DataFrame, ds_name: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(df)
    high = int((df["risk_level"] == "High Risk").sum())
    pct = (100.0 * high / total) if total > 0 else 0.0
    avg_upi_local = float(df["UPI"].mean()) if total > 0 else 0.0

    weights = (
        f"BAS={wBAS:.2f};CRS={wCRS:.2f};CARS={wCARS:.2f};"
        f"PEN={wPEN:.2f};FEI={wFEI:.2f}"
    )
    thresholds = f"high={high_thr};medium={med_thr}"

    header = [
        "timestamp",
        "model_version",
        "dataset_name",
        "total_patients",
        "high_risk_patients",
        "high_risk_pct",
        "avg_upi",
        "norm_method",
        "weights",
        "thresholds",
    ]

    row = [
        now,
        MODEL_VERSION,
        ds_name,
        total,
        high,
        f"{pct:.2f}",
        f"{avg_upi_local:.2f}",
        norm_method,
        weights,
        thresholds,
    ]

    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


# =====================================================================
# VIEW MODE SELECTOR (RADIO INSTEAD OF MANY TABS)
# =====================================================================

st.markdown("### View Mode")

mode = st.radio(
    "Select view",
    ["Executive Overview", "Provider / Network", "Patient Panel"],
    horizontal=True,
)

st.markdown("---")

# =====================================================================
# EXECUTIVE OVERVIEW
# =====================================================================

if mode == "Executive Overview":
    st.subheader("Executive Overview – Portfolio Risk & Financial Link")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='card'><h2>{total_patients}</h2><p>Total patients</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card'><h2 style='color:#d62728;'>{num_high}</h2><p>High Risk</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='card'><h2>{avg_upi:.1f}</h2><p>Average UPI</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='card card-accent'>"
            f"<h2 class='card-small-number'>{norm_method}</h2>"
            f"<p>Normalization</p></div>",
            unsafe_allow_html=True,
        )

    # Logging button
    if st.button("Log this run (append to logs/upi_runs_log.csv)"):
        log_run(data, dataset_name)
        st.success("Run logged successfully.")

    st.markdown("#### Portfolio Charts")

    # Row 1: UPI trend + Loss ratio vs UPI
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
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
            title="Average UPI by period",
            labels={"period": "Period", "UPI": "Avg UPI"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with row1_col2:
        if has_premium:
            tmp = (
                data.groupby("period")
                .agg(
                    avg_upi=("UPI", "mean"),
                    claims=("claims_amount", "sum"),
                    premium=("premium_amount", "sum"),
                )
                .reset_index()
            )
            tmp["loss_ratio"] = 100.0 * tmp["claims"] / (tmp["premium"] + 1e-9)
            fig_lr = px.line(
                tmp,
                x="period",
                y=["avg_upi", "loss_ratio"],
                markers=True,
                title="Average UPI vs Loss Ratio",
                labels={"value": "Value", "period": "Period", "variable": "Metric"},
            )
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Column 'premium_amount' not found – loss ratio cannot be computed.")

    # Row 2: Region + Top providers
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("##### Average UPI by region")
        if has_region:
            reg = data.groupby("region")["UPI"].mean().reset_index()
            fig_reg = px.bar(
                reg,
                x="region",
                y="UPI",
                title="Average UPI by region",
                labels={"region": "Region", "UPI": "Avg UPI"},
                color="UPI",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("Add 'region' column to see regional risk.")

    with row2_col2:
        st.markdown("##### Top 10 providers by average UPI")
        prov = (
            data.groupby("provider_name")["UPI"]
            .mean()
            .reset_index()
            .sort_values("UPI", ascending=False)
            .head(10)
        )
        fig_prov = px.bar(
            prov,
            x="provider_name",
            y="UPI",
            title="Top 10 providers by Avg UPI",
            labels={"provider_name": "Provider", "UPI": "Avg UPI"},
            color="UPI",
            color_continuous_scale="Plasma",
        )
        st.plotly_chart(fig_prov, use_container_width=True)
        st.dataframe(prov, use_container_width=True)

# =====================================================================
# PROVIDER / NETWORK VIEW
# =====================================================================

elif mode == "Provider / Network":
    st.subheader("Provider / Network Performance")

    providers = sorted(data["provider_name"].unique())
    selected_provider = st.selectbox("Select provider", providers)

    d = data[data["provider_name"] == selected_provider]

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(d))
    c2.metric("High-risk", int((d["risk_level"] == "High Risk").sum()))
    c3.metric("Avg UPI", f"{float(d['UPI'].mean()):.1f}")

    st.markdown("#### Provider Risk Profile")

    # Row 1: Risk composition + Provider vs network
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        comp = d[["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]].mean().reset_index()
        comp.columns = ["Score", "Value"]
        fig_comp = px.bar(
            comp,
            x="Score",
            y="Value",
            title=f"Risk composition – {selected_provider}",
            labels={"Score": "Component", "Value": "Mean value"},
            color="Value",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with row1_col2:
        col_a, col_b = st.columns(2)
        col_a.metric("Provider Avg UPI", f"{float(d['UPI'].mean()):.1f}")
        col_b.metric("Network Avg UPI", f"{avg_upi:.1f}")

        drg_prov = (
            d.groupby("drg_group")["UPI"]
            .mean()
            .reset_index()
            .sort_values("UPI", ascending=False)
        )
        fig_drg_prov = px.bar(
            drg_prov,
            x="drg_group",
            y="UPI",
            title=f"Average UPI by DRG – {selected_provider}",
            labels={"drg_group": "DRG", "UPI": "Avg UPI"},
            color="UPI",
            color_continuous_scale="Plasma",
        )
        st.plotly_chart(fig_drg_prov, use_container_width=True)

    st.markdown("#### Time Series – DRG & PCS")

    # Row 2: DRG trend + PCS trend
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        drg_trend = (
            d.groupby(["period", "drg_group"])["UPI"]
            .mean()
            .reset_index()
            .sort_values(["period", "drg_group"])
        )
        fig_drg_trend = px.line(
            drg_trend,
            x="period",
            y="UPI",
            color="drg_group",
            markers=True,
            title="Avg UPI by DRG over time",
            labels={"period": "Period", "UPI": "Avg UPI", "drg_group": "DRG"},
        )
        st.plotly_chart(fig_drg_trend, use_container_width=True)

    with row2_col2:
        pcs_trend = (
            d.groupby("period")["PCS"]
            .mean()
            .reset_index()
            .sort_values("period")
        )
        fig_pcs = px.line(
            pcs_trend,
            x="period",
            y="PCS",
            markers=True,
            title="PCS trend (documentation/compliance)",
            labels={"period": "Period", "PCS": "PCS"},
        )
        st.plotly_chart(fig_pcs, use_container_width=True)

    # Network-level DRG heatmap (for context)
    st.markdown("#### Network DRG Risk Heatmap")

    matrix = (
        data.groupby(["provider_name", "drg_group"])["UPI"]
        .mean()
        .reset_index()
    )
    pivot = matrix.pivot(index="provider_name", columns="drg_group", values="UPI")
    st.dataframe(pivot, use_container_width=True)

    fig_hm = px.imshow(
        pivot,
        aspect="auto",
        labels={"x": "DRG", "y": "Provider", "color": "Avg UPI"},
        title="Provider × DRG – Average UPI heatmap (network)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# =====================================================================
# PATIENT PANEL
# =====================================================================

elif mode == "Patient Panel":
    st.subheader("Patient Risk Panel & Timeline")

    patients = sorted(data["patient_id"].unique())
    selected_patient = st.selectbox("Select patient", patients)

    pt = data[data["patient_id"] == selected_patient].copy()
    pt = pt.sort_values("period")
    latest = pt.iloc[-1]

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("UPI (latest)", f"{float(latest['UPI']):.1f}")
    c2.metric("Risk level", latest["risk_level"])
    c3.metric("Provider (latest)", str(latest["provider_name"]))

    st.markdown("#### A) Latest Risk Drivers")

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
        title="Risk drivers (latest)",
        labels={"Score": "Component", "Value": "Score value"},
        color="Value",
        color_continuous_scale="Sunset",
    )
    st.plotly_chart(fig_drv, use_container_width=True)

    c4, c5 = st.columns(2)
    c4.metric("PCS (latest)", f"{float(latest['PCS']):.1f}")
    c5.metric("Provider penalty (latest)", f"{float(latest['provider_penalty']):.1f}")

    st.markdown("#### B) Risk Timeline (UPI + components)")

    if len(pt) >= 2:
        timeline_metrics = ["UPI", "BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]
        timeline_df = pt[["period"] + timeline_metrics].copy()

        tidy = timeline_df.melt(
            id_vars="period",
            value_vars=timeline_metrics,
            var_name="Metric",
            value_name="Value",
        )

        fig_timeline = px.line(
            tidy,
            x="period",
            y="Value",
            color="Metric",
            markers=True,
            title="Risk metrics over time for selected patient",
            labels={"period": "Period", "Value": "Score value", "Metric": "Metric"},
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.caption(
            "This timeline shows how UPI and its components (BAS, CRS, CARS, PCS, PPS, FEI) "
            "evolve over time for this patient."
        )
    else:
        st.info("Only one record available for this patient – timeline requires at least 2 periods.")

    st.markdown("#### C) Financial & DRG variance Timeline")

    if len(pt) >= 2:
        fin_df = pt[["period", "claims_amount", "drg_variance"]].copy()

        row_fin1, row_fin2 = st.columns(2)

        with row_fin1:
            fig_claims = px.bar(
                fin_df,
                x="period",
                y="claims_amount",
                title="Claims amount over time",
                labels={"period": "Period", "claims_amount": "Claims amount"},
            )
            st.plotly_chart(fig_claims, use_container_width=True)

        with row_fin2:
            fig_var = px.line(
                fin_df,
                x="period",
                y="drg_variance",
                markers=True,
                title="DRG variance (%) over time",
                labels={"period": "Period", "drg_variance": "DRG variance (%)"},
            )
            st.plotly_chart(fig_var, use_container_width=True)

        st.caption(
            "These charts link the actual claims trend with the deviation from DRG-expected cost "
            "(DRG variance) for the same patient."
        )
    else:
        st.info("Financial timeline requires at least 2 records for this patient.")

# =====================================================================
# ADVANCED TOOLS (DRG / VALIDATION / LOGS / SIMULATOR)
# =====================================================================

st.markdown("---")
st.markdown("### Advanced Analytical Tools")

tab_drg, tab_val, tab_logs, tab_sim = st.tabs(
    ["DRG-Level Comparison", "Validation Lab", "Run Logs", "Business Impact Simulator"]
)

# DRG TAB
with tab_drg:
    st.subheader("DRG-Level Risk Comparison (Network)")

    st.markdown("#### 1) Average UPI by DRG")
    drg_rank = (
        data.groupby("drg_group")["UPI"]
        .mean()
        .reset_index()
        .sort_values("UPI", ascending=False)
    )
    fig_drg_rank = px.bar(
        drg_rank,
        x="drg_group",
        y="UPI",
        title="Average UPI by DRG (network)",
        labels={"drg_group": "DRG", "UPI": "Avg UPI"},
        color="UPI",
        color_continuous_scale="Inferno",
    )
    st.plotly_chart(fig_drg_rank, use_container_width=True)
    st.dataframe(drg_rank, use_container_width=True)

    st.markdown("#### 2) Average DRG variance (%) by DRG")
    drg_var_rank = (
        data.groupby("drg_group")["drg_variance"]
        .mean()
        .reset_index()
        .sort_values("drg_variance", ascending=False)
    )
    fig_drg_var = px.bar(
        drg_var_rank,
        x="drg_group",
        y="drg_variance",
        title="Average DRG variance by DRG (network)",
        labels={"drg_group": "DRG", "drg_variance": "Avg DRG variance (%)"},
        color="drg_variance",
        color_continuous_scale="Magma",
    )
    st.plotly_chart(fig_drg_var, use_container_width=True)
    st.dataframe(drg_var_rank, use_container_width=True)

# VALIDATION LAB
with tab_val:
    st.subheader("Model Validation Lab – Correlation & Calibration")

    st.markdown("#### 1) Correlation matrix")

    candidate_cols = [
        "BAS",
        "CRS",
        "CARS",
        "PCS",
        "PPS",
        "FEI",
        "UPI",
        "drg_variance",
        "claims_amount",
    ]
    corr_cols = [c for c in candidate_cols if c in data.columns]

    if len(corr_cols) >= 2:
        corr_df = data[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        st.dataframe(corr_df, use_container_width=True)
        fig_corr = px.imshow(
            corr_df,
            aspect="auto",
            text_auto=True,
            labels={"color": "Correlation"},
            title="Correlation heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute correlation matrix.")

    st.markdown("#### 2) UPI vs claims amount")

    df_sc = data.copy()
    df_sc["claims_amount"] = pd.to_numeric(df_sc["claims_amount"], errors="coerce")
    df_sc = df_sc.dropna(subset=["UPI", "claims_amount"])

    if len(df_sc) > 0:
        fig_sc = px.scatter(
            df_sc,
            x="UPI",
            y="claims_amount",
            color="provider_name",
            trendline="ols",
            title="UPI vs claims amount (with OLS trendline)",
            labels={"UPI": "UPI", "claims_amount": "Claims amount"},
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        corr_upi_claims = df_sc["UPI"].corr(df_sc["claims_amount"])
        st.caption(f"Correlation(UPI, claims) = {corr_upi_claims:.3f}")
    else:
        st.info("Not enough valid data to plot UPI vs claims.")

    st.markdown("#### 3) Provider avg UPI vs avg DRG variance")

    prov_val = (
        data.groupby("provider_name")
        .agg(
            avg_upi=("UPI", "mean"),
            avg_drg_var=("drg_variance", "mean"),
            n=("patient_id", "count"),
        )
        .reset_index()
    )

    if len(prov_val) > 0:
        fig_pv = px.scatter(
            prov_val,
            x="avg_upi",
            y="avg_drg_var",
            size="n",
            hover_name="provider_name",
            title="Provider avg UPI vs avg DRG variance",
            labels={
                "avg_upi": "Avg UPI",
                "avg_drg_var": "Avg DRG variance (%)",
                "n": "Patients",
            },
        )
        st.plotly_chart(fig_pv, use_container_width=True)
    else:
        st.info("No provider-level aggregates available.")

    st.markdown("#### 4) DRG variance distribution")
    dv = pd.to_numeric(data["drg_variance"], errors="coerce").dropna()
    if len(dv) > 0:
        fig_hist = px.histogram(
            dv,
            nbins=40,
            title="DRG variance (%) distribution",
            labels={"value": "DRG variance (%)", "count": "Count"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No DRG variance data to display.")

# LOGS
with tab_logs:
    st.subheader("Run Logs – Model Governance")

    if os.path.isfile(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = log_df.sort_values("timestamp", ascending=False)
        st.dataframe(log_df, use_container_width=True)

        if len(log_df) >= 2:
            trend_log = log_df.copy()
            fig_log_trend = px.line(
                trend_log,
                x="timestamp",
                y="avg_upi",
                markers=True,
                title="Logged runs – Average UPI over time",
                labels={"timestamp": "Timestamp", "avg_upi": "Avg UPI"},
            )
            st.plotly_chart(fig_log_trend, use_container_width=True)
    else:
        st.info("No log file found yet. Use 'Log this run' in Executive view to create it.")

# SIMULATOR
with tab_sim:
    st.subheader("Business Impact Simulator (What-if Analysis)")

    st.write(
        "⚠️ This tab uses hypothetical improvements (PCS/PPS) on the current dataset. "
        "Outputs are for scenario illustration only."
    )

    scope = st.radio(
        "Scope of improvement",
        ["All providers", "Single provider"],
        horizontal=True,
        key="sim_scope",
    )

    if scope == "Single provider":
        prov_list = sorted(data["provider_name"].unique())
        selected_scope_provider = st.selectbox(
            "Select provider for simulation", prov_list, key="sim_provider"
        )
        mask_scope = data["provider_name"] == selected_scope_provider
    else:
        selected_scope_provider = None
        mask_scope = pd.Series(True, index=data.index)

    st.markdown("#### Choose improvement scenario")

    improve_pcs = st.checkbox("Improve PCS", value=True, key="sim_pcs")
    delta_pcs = (
        st.slider("PCS improvement (points)", 0.0, 20.0, 5.0, key="sim_pcs_delta")
        if improve_pcs
        else 0.0
    )

    improve_pps = st.checkbox("Improve PPS (performance)", value=False, key="sim_pps")
    delta_pps = (
        st.slider("PPS improvement (points)", 0.0, 20.0, 5.0, key="sim_pps_delta")
        if improve_pps
        else 0.0
    )

    if not improve_pcs and not improve_pps:
        st.info("Select at least one improvement (PCS or PPS) to run the simulation.")
    else:
        base = data.copy()
        base_high = int((base["risk_level"] == "High Risk").sum())
        base_avg_upi = float(base["UPI"].mean())

        sim = data.copy()

        if improve_pcs:
            sim.loc[mask_scope, "PCS"] = (sim.loc[mask_scope, "PCS"] + delta_pcs).clip(
                upper=100.0
            )
        if improve_pps:
            sim.loc[mask_scope, "PPS"] = (sim.loc[mask_scope, "PPS"] + delta_pps).clip(
                upper=100.0
            )

        sim["provider_penalty"] = sim.apply(calc_provider_penalty, axis=1)
        sim["UPI"] = sim.apply(calc_upi, axis=1)
        sim["risk_level"] = sim["UPI"].apply(classify_risk)

        sim_high = int((sim["risk_level"] == "High Risk").sum())
        sim_avg_upi = float(sim["UPI"].mean())

        st.markdown("#### Results – Before vs After")

        c1, c2, c3 = st.columns(3)
        c1.metric("High-risk patients (baseline)", base_high)
        c2.metric(
            "High-risk patients (simulation)",
            sim_high,
            f"{sim_high - base_high:+d}",
        )
        c3.metric(
            "Average UPI (baseline → simulation)",
            f"{base_avg_upi:.1f}",
            f"{sim_avg_upi - base_avg_upi:+.1f}",
        )

        st.markdown("#### Risk distribution – baseline vs simulation")

        def risk_counts(df, label):
            ct = df["risk_level"].value_counts().reindex(
                ["High Risk", "Medium Risk", "Low Risk"], fill_value=0
            )
            out = ct.reset_index()
            out.columns = ["risk_level", "count"]
            out["scenario"] = label
            return out

        base_counts = risk_counts(base, "Baseline")
        sim_counts = risk_counts(sim, "Simulation")
        compare_df = pd.concat([base_counts, sim_counts], ignore_index=True)

        fig_risk = px.bar(
            compare_df,
            x="risk_level",
            y="count",
            color="scenario",
            barmode="group",
            title="Risk distribution before vs after improvement",
            labels={"risk_level": "Risk level", "count": "Patients"},
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("#### Top patients with largest UPI improvement")

        merged = base[["patient_id", "UPI"]].merge(
            sim[["patient_id", "UPI"]],
            on="patient_id",
            suffixes=("_base", "_sim"),
        )
        merged["delta_upi"] = merged["UPI_sim"] - merged["UPI_base"]
        improved = merged.sort_values("delta_upi").head(10)

        st.dataframe(improved, use_container_width=True)

# =====================================================================
# FOOTER SIGNATURE
# =====================================================================

st.markdown(
    f"<div class='footer'>Developed by <b>Mudather</b> • {MODEL_VERSION} • "
    + pd.Timestamp.today().strftime("%Y-%m-%d")
    + "</div>",
    unsafe_allow_html=True,
)
