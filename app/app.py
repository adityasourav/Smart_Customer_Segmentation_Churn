import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "notebook/models/churn_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "notebook/models/scaler.pkl"
    )
)

st.title(
    "Smart Customer Segmentation & Churn Prediction System"
)

tenure = st.number_input(
    "Tenure",
    min_value=0
)

monthly = st.number_input(
    "Monthly Charges"
)

total = st.number_input(
    "Total Charges"
)

if st.button("Predict Churn"):

    data = np.array(
        [[tenure,monthly,total]]
    )

    data = scaler.transform(
        data
    )

    prediction = model.predict(
        data
    )

    if prediction[0]==1:
        st.error(
            "Customer likely to churn"
        )

    else:
        st.success(
            "Customer likely to stay"
        )