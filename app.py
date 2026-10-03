import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)


# --------------------------------------------------
# Simple styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            max-width: 900px;
            padding-top: 2rem;
        }

        h1 {
            font-size: 2rem;
            font-weight: 600;
        }

        h2 {
            font-size: 1.35rem;
            font-weight: 600;
        }

        .result-box {
            padding: 1.25rem;
            border: 1px solid #d9d9d9;
            border-radius: 6px;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .result-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .probability {
            font-size: 2rem;
            font-weight: 600;
            margin: 0.25rem 0 1rem 0;
        }

        .note {
            color: #666666;
            font-size: 0.9rem;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e5e5;
            color: #777777;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = "models/heart_disease_random_forest.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Heart Disease Prediction")

st.write(
    "Enter the patient information below to obtain a prediction "
    "from the trained machine-learning model."
)

st.caption(
    "This application is an educational machine-learning demonstration "
    "and is not a medical diagnostic tool."
)

st.divider()


# --------------------------------------------------
# Input form
# --------------------------------------------------

st.subheader("Patient Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: (
                "Female (0)" if x == 0 else "Male (1)"
            )
        )

        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            )
        )

        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2]
        )

    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            )
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            options=[0, 1, 2]
        )

        ca = st.selectbox(
            "Number of Major Vessels (CA)",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thal",
            options=[0, 1, 2, 3]
        )

    st.write("")

    submitted = st.form_submit_button(
        "Run Prediction",
        width="stretch"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }]
    )

    prediction = model.predict(input_data)[0]

    probability_class_1 = model.predict_proba(
        input_data
    )[0, 1]

    probability_class_0 = 1 - probability_class_1


    # --------------------------------------------------
    # Interpretation level
    # --------------------------------------------------

    if probability_class_1 < 0.30:

        result_level = "Level 1"
        result_title = "Lower model likelihood"

        result_message = (
            "The model estimates a lower likelihood of heart disease "
            "based on the information provided."
        )

        result_note = (
            "This does not rule out heart disease. If you have symptoms "
            "or health concerns, consider discussing them with a doctor."
        )

    elif probability_class_1 < 0.70:

        result_level = "Level 2"
        result_title = "Intermediate model likelihood"

        result_message = (
            "The model result falls within an intermediate range, "
            "so the prediction is less certain."
        )

        result_note = (
            "Consider discussing the result with a doctor, especially "
            "if you have symptoms or other concerns."
        )

    else:

        result_level = "Level 3"
        result_title = "Higher model likelihood"

        result_message = (
            "The model estimates a higher likelihood of heart disease "
            "based on the information provided."
        )

        result_note = (
            "This result does not confirm that you have heart disease. "
            "Consider discussing the result with a doctor for a proper "
            "medical evaluation."
        )


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div class="result-box">

            <div class="result-title">
                {result_level}: {result_title}
            </div>

            <p>
                {result_message}
            </p>

            <div class="probability">
                {probability_class_1:.1%}
            </div>

            <p>
                Estimated probability of Class 1
            </p>

            <p class="note">
                {result_note}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # Probability breakdown
    # --------------------------------------------------

    st.subheader("Probability")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Class 1",
            f"{probability_class_1:.1%}"
        )

    with result_col2:
        st.metric(
            "Class 0",
            f"{probability_class_0:.1%}"
        )


    st.progress(
        probability_class_1,
        text=f"Class 1 probability: {probability_class_1:.1%}"
    )


    # --------------------------------------------------
    # Submitted information
    # --------------------------------------------------

    with st.expander("View submitted information"):

        display_data = input_data.T.rename(
            columns={0: "Value"}
        )

        st.dataframe(
            display_data,
            width="stretch"
        )


    # --------------------------------------------------
    # Technical information
    # --------------------------------------------------

    with st.expander("Technical information"):

        st.write(
            f"Predicted class: {prediction}"
        )

        st.write(
            f"Class 1 probability: {probability_class_1:.4f}"
        )

        st.write(
            f"Class 0 probability: {probability_class_0:.4f}"
        )

        st.write(
            "Model: Tuned Random Forest"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Heart Disease Machine Learning Project<br>
        Random Forest classification model
    </div>
    """,
    unsafe_allow_html=True
)