import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# ---------------- UI SETTINGS ----------------
st.set_page_config(page_title="Patient Classification", layout="wide")

st.title("🏥 Hospital Patient Classification Dashboard")
st.markdown("### Outlier Detection using Isolation Forest")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ---------------- LOAD DATA ----------------
try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Uploaded successfully")
    else:
        st.sidebar.info("Using default dataset")
        data = pd.read_csv("heart_disease_uci.csv")

    st.subheader("📊 Raw Dataset")
    st.dataframe(data.head())

    # ---------------- PREPROCESS ----------------
    if 'target' in data.columns:
        data = data.drop(columns=['target'])

    data = data.dropna()

    # 🔥 IMPORTANT FIX (for deployment)
    data_processed = pd.get_dummies(data)
    data_processed = data_processed.select_dtypes(include=['number'])

    st.success("✅ Data Preprocessed Successfully")

    # ---------------- SIDEBAR ----------------
    st.sidebar.header("⚙️ Model Settings")
    contamination = st.sidebar.slider("Outlier Percentage", 0.01, 0.5, 0.1)

    # ---------------- VISUALIZATION ----------------
    st.subheader("📈 Feature Distribution")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=data_processed, ax=ax1)
    plt.xticks(rotation=90)
    st.pyplot(fig1)

    # ---------------- MODEL ----------------
    model = IsolationForest(contamination=contamination, random_state=42)
    data_processed['anomaly'] = model.fit_predict(data_processed)

    data_processed['status'] = data_processed['anomaly'].map({
        1: 'Normal',
        -1: 'Abnormal'
    })

    # ---------------- RESULTS ----------------
    st.subheader("🧠 Classification Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", len(data_processed))

    with col2:
        abnormal_count = len(data_processed[data_processed['status'] == 'Abnormal'])
        st.metric("Abnormal Patients", abnormal_count)

    # ---------------- COUNT PLOT ----------------
    st.subheader("📊 Patient Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x='status', data=data_processed, palette='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # ---------------- TABLE ----------------
    st.subheader("📋 Classified Data")
    st.dataframe(data_processed.head(20))

    # ---------------- DOWNLOAD ----------------
    st.subheader("⬇️ Download Results")
    csv = data_processed.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "classified_data.csv", "text/csv")

except Exception as e:
    st.error(f"❌ Error: {e}")