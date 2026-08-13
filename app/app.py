import streamlit as st 
import pandas as pd
import shap
import matplotlib.pyplot as plt 
import joblib
from pathlib import Path

st.set_page_config(page_title="Dementia Prediction System",
                   page_icon="🧠",
                   layout="wide")
#App Title
st.title("🧠 Dementia Prediction System")
st.write("Predict dementia risk using Machine Learning and Explainable AI (SHAP).")
#================================
#Sidebar
#================================

st.sidebar.title("🧠 Dementia Prediction System")
st.sidebar.markdown("---")
st.sidebar.header("📋 About Project")
st.sidebar.info("""
This application predicts the likelihood of dementia  using a trained Random Forest Machine Learning Model.

The prediction is explained using SHAP Explainable AI.
""")
st.sidebar.markdown('---')
st.sidebar.header("🤖 Model Information")
st.sidebar.success("Model : Random Forest")
st.sidebar.write("Algorithm : Random Forest Classifier")
st.sidebar.write("Explainability : SHAP")

st.sidebar.markdown("---")

st.sidebar.header("👨‍💻 Developed By")

st.sidebar.write("Ayush Kumar Singh")
st.sidebar.write("B.Tech Biotechnology")

# ==========================
# Load Trained Model
# ==========================

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

    rf_model = joblib.load(MODEL_PATH)
    st.success("✅ Model Loaded Successfully")

except Exception as e:
    st.error("❌ Unable to load the trained model.")
    st.stop()

st.header("📝 Enter Patient Details")
st.write("Fill the patient information below to predict dementia risk.")

col1, col2 = st.columns(2)

#======================================
with col1:

    age = st.number_input("Age (Years)", 60, 100, 75)

    gender = st.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender == "Female" else 1

    ethnicity = st.selectbox(
        "Ethnicity",
        ["White", "Black", "Asian", "Other"]
    )
    ethnicity = ["White", "Black", "Asian", "Other"].index(ethnicity)

    education = st.selectbox(
        "Education Level",
        ["None", "High School", "Bachelor", "Higher"]
    )
    education = ["None", "High School", "Bachelor", "Higher"].index(education)

    bmi = st.number_input("Body Mass Index (BMI)", 15.0, 45.0, 25.0)

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    smoking = 0 if smoking == "No" else 1

    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    alcohol = 0 if alcohol == "No" else 1

    physical_activity = st.slider("Physical Activity Score", 0, 10, 5)

    diet_quality = st.slider("Diet Quality Score", 0, 10, 5)

    sleep_quality = st.slider("Sleep Quality Score", 0, 10, 5)

    family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
    family_history = 0 if family_history == "No" else 1

    cardiovascular = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
    cardiovascular = 0 if cardiovascular == "No" else 1

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    diabetes = 0 if diabetes == "No" else 1

    depression = st.selectbox("Depression", ["No", "Yes"])
    depression = 0 if depression == "No" else 1

    head_injury = st.selectbox("History of Head Injury", ["No", "Yes"])
    head_injury = 0 if head_injury == "No" else 1

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    hypertension = 0 if hypertension == "No" else 1

#=====================================
with col2:

    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 220, 120)

    diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 140, 80)

    cholesterol_total = st.number_input("Total Cholesterol", 100, 400, 200)

    cholesterol_ldl = st.number_input("LDL Cholesterol", 20, 250, 100)

    cholesterol_hdl = st.number_input("HDL Cholesterol", 10, 120, 50)

    cholesterol_triglycerides = st.number_input("Triglycerides", 50, 500, 150)

    mmse = st.slider("MMSE Score", 0, 30, 20)

    functional = st.slider("Functional Assessment Score", 0, 10, 5)

    memory = st.selectbox("Memory Complaints", ["No", "Yes"])
    memory = 0 if memory == "No" else 1

    behavioral = st.selectbox("Behavioral Problems", ["No", "Yes"])
    behavioral = 0 if behavioral == "No" else 1

    adl = st.slider("Activities of Daily Living (ADL)", 0, 10, 5)

    confusion = st.selectbox("Confusion", ["No", "Yes"])
    confusion = 0 if confusion == "No" else 1

    disorientation = st.selectbox("Disorientation", ["No", "Yes"])
    disorientation = 0 if disorientation == "No" else 1

    personality = st.selectbox("Personality Changes", ["No", "Yes"])
    personality = 0 if personality == "No" else 1

    difficulty = st.selectbox("Difficulty Completing Daily Tasks", ["No", "Yes"])
    difficulty = 0 if difficulty == "No" else 1

    forgetfulness = st.selectbox("Forgetfulness", ["No", "Yes"])
    forgetfulness = 0 if forgetfulness == "No" else 1

#====================================
#Predict 
#====================================
predict_button = st.button("🔍 Predict Dementia")
if predict_button:
    if age < 60 or age > 100:
        st.error("❌ Age must be between 60 and 100 years.")
        st.stop()

    if bmi < 15 or bmi > 45:
        st.error("❌ Please enter a valid BMI between 15 and 45.")
        st.stop()

    if systolic_bp < 80 or systolic_bp > 220:
        st.error("❌ Please enter a valid systolic blood pressure.")
        st.stop()

    if diastolic_bp < 40 or diastolic_bp > 140:
        st.error("❌ Please enter a valid diastolic blood pressure.")
        st.stop()

    new_patient = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Ethnicity": [ethnicity],
        "EducationLevel": [education],
        "BMI": [bmi],
        "Smoking": [smoking],
        "AlcoholConsumption": [alcohol],
        "PhysicalActivity": [physical_activity],
        "DietQuality": [diet_quality],
        "SleepQuality": [sleep_quality],
        "FamilyHistoryAlzheimers": [family_history],
        "CardiovascularDisease": [cardiovascular],
        "Diabetes": [diabetes],
        "Depression": [depression],
        "HeadInjury": [head_injury],
        "Hypertension": [hypertension],
        "SystolicBP": [systolic_bp],
        "DiastolicBP": [diastolic_bp],
        "CholesterolTotal": [cholesterol_total],
        "CholesterolLDL": [cholesterol_ldl],
        "CholesterolHDL": [cholesterol_hdl],
        "CholesterolTriglycerides": [cholesterol_triglycerides],
        "MMSE": [mmse],
        "FunctionalAssessment": [functional],
        "MemoryComplaints": [memory],
        "BehavioralProblems": [behavioral],
        "ADL": [adl],
        "Confusion": [confusion],
        "Disorientation": [disorientation],
        "PersonalityChanges": [personality],
        "DifficultyCompletingTasks": [difficulty],
        "Forgetfulness": [forgetfulness]
    })


    try:
        prediction = rf_model.predict(new_patient)
        probability = rf_model.predict_proba(new_patient)

    except Exception as e:
        st.error("❌ Unable to generate prediction.")
        st.stop()
    

#====================================
#Result
#====================================

    st.markdown("---")
    st.header("🩺 Prediction Result")

    if prediction[0] == 1:
        st.error(
        "🔴 **Dementia Risk Detected**\n\n"
        "The model predicts a higher likelihood of dementia."
        )
    else:
        st.success(
        "🟢 **Low Dementia Risk Detected**\n\n"
        "The model does not predict dementia for this patient."
        )

# Probability of Dementia
    dementia_probability = probability[0][1] * 100

# Confidence of Prediction
    prediction_confidence = max(probability[0]) * 100

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
        label="🎯 Prediction Confidence",
        value=f"{prediction_confidence:.2f}%"
        )

    with metric2:
        st.metric(
        label="🧠 Dementia Probability",
        value=f"{dementia_probability:.2f}%"
        )


#==========================
# Risk Assessment
# ==========================

    st.subheader("📊 Risk Assessment")

    if dementia_probability < 30:
        risk_level = "🟢 Low Risk"
        risk_message = "The model estimates a relatively low probability of dementia."
        st.success(f"**{risk_level}**\n\n{risk_message}")

    elif dementia_probability < 70:
        risk_level = "🟡 Moderate Risk"
        risk_message = "The model estimates an intermediate probability of dementia. Further assessment may be appropriate."
        st.warning(f"**{risk_level}**\n\n{risk_message}")

    else:
        risk_level = "🔴 High Risk"
        risk_message = "The model estimates a high probability of dementia. Professional clinical evaluation is recommended."
        st.error(f"**{risk_level}**\n\n{risk_message}")

#Disclaimer
    st.markdown("---")

    st.info("""
⚠️ **Medical Disclaimer**

This application is developed for educational and research purposes only.

The prediction generated by this machine learning model should **not be considered a medical diagnosis**.

Please consult a qualified healthcare professional for clinical evaluation and medical decisions.
""")
    st.markdown("---")
    st.caption(
        "Dementia Prediction System | Random Forest + SHAP Explainable AI"
    )
#===============================
#SHAP 
#===============================
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(new_patient)
    st.markdown("---")
    st.subheader("🔍 SHAP Waterfall Explanation")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(
        shap_values[0, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)

#=============================
#SHAP Feature Importance
#=============================
    st.markdown("---")
    st.subheader("📊 SHAP Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)

#===============================
#SHAP Beeswarm plot
#===============================
    st.markdown("---")
    st.subheader("🐝 SHAP Beeswarm Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(
    shap_values[:, :, 1],
    max_display=15,
    show=False
    )
    st.pyplot(fig)
    plt.close(fig)