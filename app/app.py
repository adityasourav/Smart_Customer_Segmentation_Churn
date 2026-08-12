import streamlit as st
import joblib
import pandas as pd
import os


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "notebook",
    "models",
    "churn_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "notebook",
    "models",
    "scaler.pkl"
)

SEGMENTATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "notebook",
    "models",
    "segmentation_model.pkl"
)

SEGMENTATION_SCALER_PATH = os.path.join(
    BASE_DIR,
    "notebook",
    "models",
    "segmentation_scaler.pkl"
)


@st.cache_resource
def load_models():

    churn_model = joblib.load(
        MODEL_PATH
    )

    churn_scaler = joblib.load(
        SCALER_PATH
    )

    segmentation_model = joblib.load(
        SEGMENTATION_MODEL_PATH
    )

    segmentation_scaler = joblib.load(
        SEGMENTATION_SCALER_PATH
    )

    return (
        churn_model,
        churn_scaler,
        segmentation_model,
        segmentation_scaler
    )


try:

    (
        model,
        scaler,
        segmentation_model,
        segmentation_scaler
    ) = load_models()

except Exception as e:

    st.error(
        "Unable to load model files."
    )

    st.code(str(e))

    st.stop()


st.title(
    "Smart Customer Churn Prediction System"
)

st.markdown(
    "Predict customer churn probability and identify "
    "customer segments using machine learning."
)

st.divider()


st.subheader(
    "Customer Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )


with col2:

    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=1.0
    )


with col3:

    total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=100000.0,
        value=840.0,
        step=10.0
    )


st.divider()


if st.button(
    "🔍 Analyze Customer",
    use_container_width=True
):

    data = pd.DataFrame(
        [[tenure, monthly, total]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    )


    # -------------------------
    # Churn Prediction
    # -------------------------

    churn_scaled = scaler.transform(
        data
    )

    prediction = model.predict(
        churn_scaled
    )

    probability = model.predict_proba(
        churn_scaled
    )

    churn_probability = (
        probability[0][1] * 100
    )

    stay_probability = (
        probability[0][0] * 100
    )


    if churn_probability >= 70:

        risk_level = "High"

    elif churn_probability >= 40:

        risk_level = "Medium"

    else:

        risk_level = "Low"


    st.subheader(
        "Churn Analysis"
    )


    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )


    with col2:

        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )


    with col3:

        st.metric(
            "Churn Risk Level",
            risk_level
        )


    st.write("")


    st.markdown(
        "**Churn Probability**"
    )


    st.progress(
        min(
            int(churn_probability),
            100
        )
    )


    st.caption(
        "This bar visualizes the likelihood of the "
        "customer churning. Higher values indicate "
        "greater churn risk."
    )


    if risk_level == "High":

        st.warning(
            "High churn risk. Consider taking proactive "
            "customer retention measures."
        )

    elif risk_level == "Medium":

        st.info(
            "Moderate churn risk. The customer may require "
            "additional attention."
        )

    else:

        st.success(
            "Low churn risk. The customer is likely to remain."
        )


    st.divider()


    # -------------------------
    # Customer Segmentation
    # -------------------------

    st.subheader(
        "Customer Segmentation"
    )


    segmentation_scaled = segmentation_scaler.transform(
        data
    )

    cluster = segmentation_model.predict(
        segmentation_scaled
    )[0]


    segment_names = {

        0: "Long-term, Low-spend",

        1: "High-value, Long-term",

        2: "New, Low-spend",

        3: "New, High-spend"
    }


    segment_name = segment_names.get(
        cluster,
        f"Customer Segment {cluster}"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Customer Segment",
            segment_name
        )


    with col2:

        st.metric(
            "Cluster",
            f"Cluster {cluster}"
        )


    st.write("")


    st.markdown(
        f"""
        **Customer Profile**

        This customer belongs to the **{segment_name}**
        segment based on tenure, monthly charges and
        total charges.
        """
    )


with st.sidebar:

    st.header(
        "About the Project"
    )


    st.write(
        "This application combines machine learning "
        "based churn prediction and customer segmentation."
    )


    st.subheader(
        "Model Inputs"
    )


    st.write(
        """
        • Tenure

        • Monthly Charges

        • Total Charges
        """
    )


    st.subheader(
        "Churn Risk Levels"
    )


    st.write(
        """
        🟢 **Low:** 0% – 39.99%

        🟠 **Medium:** 40% – 69.99%

        🔴 **High:** 70% – 100%
        """
    )


    st.subheader(
        "Customer Segments"
    )


    st.write(
        """
        **Cluster 0:** Long-term, Low-spend

        **Cluster 1:** High-value, Long-term

        **Cluster 2:** New, Low-spend

        **Cluster 3:** New, High-spend
        """
    )


    st.subheader(
        "Technology"
    )


    st.write(
        """
        • Python

        • Scikit-learn

        • Pandas

        • Streamlit
        """
    )


    st.divider()


    st.caption(
        "Smart Customer Segmentation & Churn Prediction System"
    )