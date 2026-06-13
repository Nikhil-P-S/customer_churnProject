import streamlit as st
import pandas as pd
import mlflow.sklearn
import os
from datetime import datetime
import plotly.express as px


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Customer Churn App",
    layout="wide"
)

st.title("Customer Churn Prediction + Real-Time Monitoring")


# =========================
# Load model
# =========================
model = mlflow.sklearn.load_model("saved_model")


# =========================
# Sidebar menu
# =========================
menu = st.sidebar.selectbox(
    "Select Page",
    ["Prediction", "Monitoring Dashboard"]
)


# =========================
# Prediction Page
# =========================
if menu == "Prediction":

    st.header("Customer Churn Prediction Form")

    CustomerID = st.number_input("Customer ID", min_value=1, value=2)
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Gender = st.selectbox("Gender", ["Female", "Male"])

    Tenure = st.number_input("Tenure", min_value=0, value=39)
    Usage_Frequency = st.number_input("Usage Frequency", min_value=0, value=14)
    Support_Calls = st.number_input("Support Calls", min_value=0, value=5)
    Payment_Delay = st.number_input("Payment Delay", min_value=0, value=18)

    Subscription_Type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    Contract_Length = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

    Total_Spend = st.number_input("Total Spend", min_value=0.0, value=932.0)
    Last_Interaction = st.number_input("Last Interaction", min_value=0, value=17)

    if st.button("Predict Churn"):

        input_df = pd.DataFrame([{
            "CustomerID": CustomerID,
            "Age": Age,
            "Gender": Gender,
            "Tenure": Tenure,
            "Usage Frequency": Usage_Frequency,
            "Support Calls": Support_Calls,
            "Payment Delay": Payment_Delay,
            "Subscription Type": Subscription_Type,
            "Contract Length": Contract_Length,
            "Total Spend": Total_Spend,
            "Last Interaction": Last_Interaction
        }])

        prediction = model.predict(input_df)
        result = int(prediction[0])

        if result == 1:
            st.error("Prediction: Customer will churn")
        else:
            st.success("Prediction: Customer will not churn")

        # Save prediction log
        log_data = input_df.copy()
        log_data["Prediction"] = result
        log_data["Prediction Meaning"] = (
            "Customer will churn" if result == 1 else "Customer will not churn"
        )
        log_data["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs("logs", exist_ok=True)
        log_file = "logs/prediction_logs.csv"

        if not os.path.exists(log_file):
            log_data.to_csv(log_file, index=False)
        else:
            log_data.to_csv(log_file, mode="a", header=False, index=False)

        st.info("Prediction saved to real-time logs")


# =========================
# Monitoring Dashboard
# =========================
elif menu == "Monitoring Dashboard":

    st.header("Real-Time Monitoring Dashboard")

    log_file = "logs/prediction_logs.csv"

    if not os.path.exists(log_file):
        st.warning("No prediction logs found. Make predictions first.")
    else:
        df = pd.read_csv(log_file)

        total_predictions = len(df)
        churn_count = (df["Prediction"] == 1).sum()
        non_churn_count = (df["Prediction"] == 0).sum()
        churn_rate = df["Prediction"].mean() * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Predictions", total_predictions)
        col2.metric("Churn Count", churn_count)
        col3.metric("Non-Churn Count", non_churn_count)
        col4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.subheader("Prediction Distribution")

        prediction_counts = df["Prediction Meaning"].value_counts().reset_index()
        prediction_counts.columns = ["Prediction", "Count"]

        fig1 = px.bar(
            prediction_counts,
            x="Prediction",
            y="Count",
            title="Churn vs Non-Churn Predictions"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Churn Trend Over Time")

        df["Prediction Time"] = pd.to_datetime(df["Prediction Time"])
        df["Date"] = df["Prediction Time"].dt.date

        daily_churn = df.groupby("Date")["Prediction"].mean().reset_index()
        daily_churn["Churn Rate"] = daily_churn["Prediction"] * 100

        fig2 = px.line(
            daily_churn,
            x="Date",
            y="Churn Rate",
            markers=True,
            title="Daily Churn Rate Trend"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Feature Monitoring")

        feature_cols = [
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction"
        ]

        avg_features = df[feature_cols].mean().reset_index()
        avg_features.columns = ["Feature", "Average Value"]

        st.dataframe(avg_features)

        st.subheader("Recent Predictions")
        st.dataframe(df.tail(20))