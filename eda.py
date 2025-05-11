import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Page config
st.set_page_config(page_title="EDA Web App", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-color: #f5f7fa;
   
    /* Dataframes */
    .stDataFrame {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 8px;
    }

    /* Button */
    .stButton > button {
        background-color: #003366;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 8px 20px;
    }
    .stButton > button:hover {
        background-color: #0059b3;
        transition: background-color 0.3s ease;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 8px 20px;
    }
    /* Headings */
    h1, h2, h3,h4{
        color: #00274d;
        font-weight: 700;
    }

    /* Info box */
    .stAlert {
        background-color: #e3f2fd;
    }
    .css-1r6s2p7 {
        color: white !important;
    }
      .stSelectbox label {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Exploratory Data Analysis App")

# Upload CSV
st.sidebar.header("📁 Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read data
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Preview Dataset")
        st.dataframe(df.head())

        # Shape
        st.markdown(f"**🧾 Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

        # Summary
        st.subheader("📈 Summary Statistics")
        st.write(df.describe())

        # Data Types
        st.subheader("🧬 Data Types")
        st.write(df.dtypes)

        # Missing Values
        st.subheader("❌ Missing Values")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])

        # Univariate Analysis
        st.subheader("📊 Univariate Analysis")
        selected_col = st.selectbox("Select a column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)

        # Bivariate Analysis
        st.subheader("🔁 Bivariate Analysis")
        col1 = st.selectbox("X-axis", df.select_dtypes(include=np.number).columns, key="x")
        col2 = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns, key="y")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df[col1], y=df[col2], ax=ax2)
        st.pyplot(fig2)

        # Correlation Matrix
        st.subheader("🔗 Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # Handling non-numeric columns for correlation
        df_numeric = df.select_dtypes(include=[np.number])
        sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        # Download processed data
        st.subheader("⬇️ Download Cleaned Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

else:
    st.info("📌 Upload a CSV file to start EDA.")
