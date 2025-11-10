import io
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# 1. CONFIG & PAGE SETUP
# =========================
st.set_page_config(
    page_title="Clinical–Actuarial Profiling Dashboard",
    layout="wide"
)

st.title("Clinical–Actuarial Scoring & Risk Dashboard")

st.markdown(
    """
    Upload a CSV or Excel file with patient scores to compute the Unified Patient Indicator (UPI),
    evaluate provider penalties, and classify patients by risk.
    """
)

# =========================
# 2. REQUIRED COLUMNS
# =========================
REQUIRED_COLUMNS = [
    "patient_id",
    "BAS",
    "CRS",
    "CARS",
    "PCS",
    "PPS",
    "FEI",
]

# =========================
# 3. HELPER FUNCTIONS
# =========================
def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def normalize_to_0_100(series: pd.Series) -> pd.Series:
    """Normalize numeric values to 0–100 range if needed."""
    if series.min() >= 0 and series.max() <= 100:
        return series
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series([100] * len(series), index=series.index)
    return (series - smin) / (smax - smin) * 100

def calc_provider_penalty(row) -> float:
    """provider_penalty = 0.6 * (100 - PCS) + 0.4 * (100 - PPS)"""
    return 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS"])

def calc_upi(row, w_bas, w_crs, w_cars, w_penalty, w_fei) -> float:
    """UPI = weighted combination of component scores"""
    return (
        w_bas * row["BAS"]
        + w_crs * row["CRS"]
        + w_cars * row["CARS"]
        + w_penalty * row["provider_penalty"]
        + w_fei * row["FEI"]
    )

def classify_patient(upi: float) -> str:
    if upi >= 80:
        return "High Risk"
    elif upi >= 60:
        return "Medium Risk"
    else:
        return "Low Risk"

# =========================
# 4. SIDEBAR: WEIGHTS
# =========================
st.sidebar.header("Weights Configuration")
st.sidebar.markdown("Adjust weights (should sum ≈ 1.0)")

w_bas = st.sidebar.slider("Weight: BAS", 0.0, 1.0, 0.25, 0.01)
w_crs = st.sidebar.slider("Weight: CRS", 0.0, 1.0, 0.25, 0.01)
w_cars = st.sidebar.slider("Weight: CARS", 0.0, 1.0, 0.25, 0.01)
w_penalty = st.sidebar.slider("Weight: Provider Penalty", 0.0, 1.0, 0.15, 0.01)
w_fei = st.sidebar.slider("Weight: FEI", 0.0, 1.0, 0.10, 0.01)

total_w = w_bas + w_crs + w_cars + w_penalty + w_fei
if abs(total_w - 1.0) > 0.001:
    st.sidebar.warning(f"Current total weight = {total_w:.2f}. Adjust to ~1.00 for accuracy.")

# =========================
# 5. FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Please upload a data file to start.")
    st.stop()

# Read file
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# =========================
# 6. VALIDATE COLUMNS
# =========================
missing_cols = check_columns(df)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# =========================
# 7. NORMALIZE SCORES
# =========================
data = df.copy()
for col in ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = normalize_to_0_100(data[col])
data = data.dropna(subset=["patient_id"])

# =========================
# 8. CALCULATIONS
# =========================
data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)
data["UPI"] = data.apply(lambda r: calc_upi(r, w_bas, w_crs, w_cars, w_penalty, w_fei), axis=1)
data["risk_level"] = data["UPI"].apply(classify_patient)

# =========================
# 9. KPIs
# =========================
total_patients = len(data)
high_risk = (data["risk_level"] == "High Risk").sum()
avg_upi = data["UPI"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("Total Patients", f"{total_patients}")
k2.metric("High-Risk Patients", f"{high_risk}")
k3.metric("Average UPI", f"{avg_upi:.2f}")

# =========================
# 10. CHART
# =========================
st.subheader("UPI Distribution")

fig = px.histogram(
    data,
    x="UPI",
    nbins=20,
    title="UPI Histogram",
    labels={"UPI": "Unified Patient Indicator"}
)
st.plotly_chart(fig, width="stretch")

# =========================
# 10b. UPI EXPLANATION
# =========================
with st.expander("ℹ️ What is the Unified Patient Indicator (UPI)?"):
    st.markdown("""
    **Unified Patient Indicator (UPI)** is a composite score (0–100) combining:
    - **BAS:** Behavioral Adherence  
    - **CRS:** Clinical Risk  
    - **CARS:** Clinical–Actuarial Risk  
    - **Provider Penalty:** based on PCS & PPS  
    - **FEI:** Financial Efficiency Index  

    **Formula:**  
    `UPI = 0.25×BAS + 0.25×CRS + 0.25×CARS + 0.15×ProviderPenalty + 0.10×FEI`

    **Interpretation:**  
    - ≥80 → **High Risk**  
    - 60–79 → **Medium Risk**  
    - <60 → **Low Risk**

    Adjust weights from the sidebar to simulate different actuarial models.
    """)

# =========================
# 11. DATA TABLE
# =========================
st.subheader("Patient-Level Results")

display_cols = [
    "patient_id",
    "BAS",
    "CRS",
    "CARS",
    "PCS",
    "PPS",
    "FEI",
    "provider_penalty",
    "UPI",
    "risk_level",
]
st.dataframe(data[display_cols], width="stretch")

# =========================
# 12. DOWNLOAD
# =========================
st.subheader("Download Results")
csv_buffer = io.StringIO()
data.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download as CSV",
    data=csv_buffer.getvalue(),
    file_name="upi_results.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Run locally: `streamlit run app.py`  |  Developed for clinical–actuarial risk profiling.")
