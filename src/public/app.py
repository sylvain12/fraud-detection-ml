import time
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from fraud_detection.config import DATA_PICKLE_PATH, SRC_DIR
from fraud_detection.ml.registry import load_model
from fraud_detection.utils import get_predictions_results, load_pickle_dataset

transactions = None
predictions_result = None
predictions_result_styled = None
predictions: list[int] = []
total_transaction = len(predictions)
transaction_counter = Counter(transactions)
total_non_fraud_transaction = 0
total_fraud_transaction = 0
is_predict = False


def get_predictions_details(predictions: list[int] = []) -> tuple:
    total_transaction = len(predictions)
    transaction_counter = Counter(predictions)
    total_non_fraud_transaction = transaction_counter.get(0, 0)
    total_fraud_transaction = transaction_counter.get(1, 0)

    return total_transaction, total_non_fraud_transaction, total_fraud_transaction


# Configuration
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="auto",
)

with open(f"{SRC_DIR}/public/style.css") as css:
    st.markdown(
        f"""
                <style>
                    {css.read()}
                </style>""",
        unsafe_allow_html=True,
    )

model = load_model()

if model is None:
    st.error("Error: Unable to load the model", icon="🚨")
    st.stop()

# @st.cache_data
# def load_data() -> pd.DataFrame:
#     return load_pickle_dataset(DATA_PICKLE_PATH)


# data = load_data()


# Sidebar Section
with st.sidebar:
    st.html(
        """
            <h1>Payment Fraud Detection</h1>
        """
    )
    st.divider()
    uploaded_file = st.file_uploader(
        "Upload your transactions file...", type=["xlsx", "csv"]
    )

    st.html("""
            <div style="margin: 50px 0;">

            </div>
            """)
    if uploaded_file is not None:
        transactions = pd.read_csv(uploaded_file)
        st.write(f"{len(transactions)} transactions uploaded for prediction")
        is_predict = st.button("Predict", type="primary")

        if is_predict:
            st.divider()
            with st.spinner("Wait transaction prediction..."):
                time.sleep(12)

            predictions = model.predict(transactions)
            num_trans, num_non_fraud_trans, num_fraud_trans = get_predictions_details(
                predictions
            )
            total_transaction = num_trans
            total_non_fraud_transaction = num_non_fraud_trans
            total_fraud_transaction = num_fraud_trans

            predictions_result_styled, predictions_result = get_predictions_results(
                transactions, predictions
            )

if not is_predict:
    pass
else:
    # Hearder Section
    with st.container():
        col1, col2, col3 = st.columns(3, vertical_alignment="center")

        with col1:
            col1.html(f"""
                    <div style="border-top: 3px solid grey; box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px; padding: 12px; font-family: Sora;">
                        <p style="font-size: 15px; text-transform: uppercase;">Total transaction</p>
                        <span style="color: grey; font-size: 45px">{total_transaction}</span>
                    </div>
            """)

        with col2:
            col2.html(f"""
                    <div style="border-top: 3px solid green; box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px; padding: 12px; font-family: Sora;">
                        <p style="font-size: 15px; text-transform: uppercase;">Total non fraud</p>
                        <span style="color: green; font-size: 45px">{total_non_fraud_transaction}</span>
                    </div>
            """)

        with col3:
            col3.html(f"""
                    <div style="border-top: 3px solid red; box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px; padding: 12px; font-family: Sora;">
                        <p style="font-size: 15px; text-transform: uppercase;">Total fraud</p>
                        <span style="color: red; font-size: 45px">{total_fraud_transaction}</span>
                    </div>
            """)

    # Chart section
    with st.container():
        col1, col2 = st.columns(2, vertical_alignment="center")

        if predictions_result_styled is not None:
            with col1:
                fig = px.bar(
                    predictions_result,
                    x="Status",
                    title="Transaction By Status",
                    color="Status",
                )
                st.plotly_chart(fig, theme="streamlit")

            with col2:
                fraud_data_by_type = (
                    predictions_result[
                        predictions_result["Status"].str.lower() == "fraud"
                    ]
                    .groupby("type")["Status"]
                    .count()
                )
                fig = px.pie(
                    fraud_data_by_type,
                    names=fraud_data_by_type.index,
                    values="Status",
                    title="Fraud Transaction By Type",
                )
                st.plotly_chart(fig, theme=None)

    st.divider()

    # Recent transaction section
    with st.container():
        st.html("""
                <div style="margin:10px;">
                    <p style="font-size: 18px">Predictions details</p>
                </div>
                """)

        if uploaded_file is not None and predictions_result is not None:
            # view = st.radio("View status", options=["All", "Non-fraud", "Fraud"])
            # if view.lower() == "fraud":
            #     content = predictions_result.copy()
            #     content[content["Status"] == "Fraud"]
            #     st.table(content)
            st.table(predictions_result_styled)
