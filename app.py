import streamlit as st
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# ------------------------------
# ADMIN MODE CHECK
# ------------------------------
ADMIN_KEY = ""

input_key = st.sidebar.text_input("Admin Key (optional)", type="password")
IS_ADMIN = (input_key == ADMIN_KEY and input_key != "")

if not IS_ADMIN:
    # Auto-refresh every 15 minutes
    st_autorefresh(interval=15 * 60 * 1000, key="refresh")

# ------------------------------
# PAGE TITLE
# ------------------------------
st.title("🎯 Clinical–Actuarial Scoring & Risk Dashboard")

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded = st.file_uploader("Upload Patient File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded:
    ext = uploaded.name.split(".")[-1]
    if ext == "csv":
        data = pd.read_csv(uploaded)
    else:
        data = pd.read_excel(uploaded)

    # Required columns
    required_cols = [
        "patient_id", "BAS", "CRS", "CARS",
        "PCS", "PPS", "FEI", "provider_readmit_score"
    ]

    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # ------------------------------
    # NORMALIZATION
    # ------------------------------
    method = st.sidebar.selectbox(
        "Normalization Method",
        ["minmax", "zscore", "none"],
        index=0
    )

    def normalize(series):
        if method == "none":
            return series
        if method == "minmax":
            return 100 * (series - series.min()) / (series.max() - series.min() + 1e-9)
        if method == "zscore":
            z = (series - series.mean()) / (series.std() + 1e-9)
            return 50 + 10 * z
        return series

    for col in ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]:
        data[col] = normalize(data[col])

    # ------------------------------
    # PROVIDER PENALTY
    # ------------------------------
    def calc_provider_penalty(row):
        penalty = 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS"])
        return penalty

    data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)

    # ------------------------------
    # UPI SCORING
    # ------------------------------
    st.sidebar.markdown("### UPI Weights")
    wBAS = st.sidebar.slider("Weight BAS", 0.0, 1.0, 0.25)
    wCRS = st.sidebar.slider("Weight CRS", 0.0, 1.0, 0.25)
    wCARS = st.sidebar.slider("Weight CARS", 0.0, 1.0, 0.25)
    wPEN = st.sidebar.slider("Weight Provider Penalty", 0.0, 1.0, 0.15)
    wFEI = st.sidebar.slider("Weight FEI", 0.0, 1.0, 0.10)

    def calc_upi(row):
        return (
            wBAS * row["BAS"] +
            wCRS * row["CRS"] +
            wCARS * row["CARS"] +
            wPEN * row["provider_penalty"] +
            wFEI * row["FEI"]
        )

    data["UPI"] = data.apply(calc_upi, axis=1)

    # ------------------------------
    # RISK CLASSIFICATION
    # ------------------------------
    def classify(x):
        if x >= 80: return "High Risk"
        if x >= 60: return "Medium Risk"
        return "Low Risk"

    data["risk_level"] = data["UPI"].apply(classify)

    # ------------------------------
    # KPIs
    # ------------------------------
    st.subheader("📌 Key Indicators")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", len(data))
    c2.metric("High Risk Patients", sum(data["risk_level"] == "High Risk"))
    c3.metric("Average UPI", round(data["UPI"].mean(), 2))

    # ------------------------------
    # RESULTS TABLE
    # ------------------------------
    st.subheader("📄 Patient-Level Results")
    st.dataframe(data)

    # ------------------------------
    # PDF EXPORT
    # ------------------------------
    def generate_pdf(df):
        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=letter)
        pdf.setFont("Helvetica", 12)
        pdf.drawString(30, 750, "Clinical–Actuarial Dashboard Summary")
        pdf.drawString(30, 730, f"Total Patients: {len(df)}")
        pdf.drawString(30, 715, f"High Risk: {sum(df['risk_level']=='High Risk')}")
        pdf.drawString(30, 700, f"Average UPI: {round(df['UPI'].mean(), 2)}")
        pdf.showPage()
        pdf.save()
        buf.seek(0)
        return buf

    if st.button("Download PDF Summary"):
        pdf_file = generate_pdf(data)
        st.download_button(
            "Download PDF",
            data=pdf_file,
            file_name="upi_summary.pdf",
            mime="application/pdf"
        )

    # ------------------------------
    # DOWNLOAD CSV
    # ------------------------------
    st.subheader("Download Results")
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download CSV",
        csv_buffer.getvalue(),
        "upi_results.csv",
        "text/csv"
    )

# ------------------------------
# FOOTER SIGNATURE
# ------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>Mudather</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model v1.3 • " + pd.Timestamp.today().strftime("%Y-%m-%d"))
