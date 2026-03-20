import streamlit as st
import pandas as pd
import joblib

# 1. Load BOTH saved models
reg_model = joblib.load('diabetes_model.pkl')
clf_model = joblib.load('diabetes_clf_model.pkl')

if not hasattr(clf_model, 'multi_class'):
    clf_model.multi_class = 'auto'

# 2. Create a Sidebar for Navigation
st.sidebar.title("🩺 AI Medical Tools")
app_mode = st.sidebar.selectbox(
    "Choose a tool:", 
    ["HbA1c Predictor (Regression)", "Diabetes Diagnosis (Classification)"]
)

# ==========================================
# TOOL 1: REGRESSION (Your original code)
# ==========================================
if app_mode == "HbA1c Predictor (Regression)":
    st.title("📈 HbA1c Predictor")
    st.write("Enter your fasting glucose level below to predict your HbA1c.")
    
    glucose_input = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=350, value=100)
    
    if st.button("Predict HbA1c"):
        input_data = pd.DataFrame({'glucose_fasting': [glucose_input]})
        prediction = reg_model.predict(input_data)[0]
        st.success(f"**Predicted HbA1c:** {prediction:.2f}")

# ==========================================
# TOOL 2: CLASSIFICATION (The new model!)
# ==========================================
elif app_mode == "Diabetes Diagnosis (Classification)":
    st.title("🩸 Diabetes Risk Predictor")
    st.write("Enter patient vitals to predict the likelihood of diabetes.")
    
    # Use columns to lay out the input fields side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        glucose_input = st.number_input("Fasting Glucose", min_value=50, max_value=350, value=100)
        bmi_input = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        
    with col2:
        age_input = st.number_input("Age", min_value=1, max_value=120, value=45)
        bp_input = st.number_input("Systolic Blood Pressure", min_value=70, max_value=200, value=120)
    
    if st.button("Predict Diabetes Status"):
        # 1. Format the data exactly how the model expects it
        input_data = pd.DataFrame({
            'glucose_fasting': [glucose_input],
            'bmi': [bmi_input],
            'age': [age_input],
            'systolic_bp': [bp_input]
        })
        
        # 2. Get the prediction and probability
        prediction = clf_model.predict(input_data)[0]
        probability = clf_model.predict_proba(input_data)[0][1] 
        
        # 3. Display the Diagnosis Results
        st.markdown("---")
        st.subheader("📋 Diagnosis Results")
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Diabetes detected.** (AI Confidence: {probability*100:.1f}%)")
        else:
            st.success(f"✅ **Low Risk of Diabetes.** (AI Confidence: {(1-probability)*100:.1f}%)")
            
        # ==========================================
        # NEW: EXPLAINABLE AI (Feature Importance)
        # ==========================================
        st.markdown("---")
        st.subheader("🧠 Why did the AI make this decision?")
        st.write("Here is how much weight the AI gave to each of your metrics:")
        
        # Extract the learned 'weights' (coefficients) from the Logistic Regression model
        # We use absolute values because we just care about the *strength* of the importance
        import numpy as np
        feature_importance = np.abs(clf_model.coef_[0])
        
        # Create a tiny dataframe just for the chart
        importance_df = pd.DataFrame({
            'Metric': ['Fasting Glucose', 'BMI', 'Age', 'Blood Pressure'],
            'Importance Score': feature_importance
        }).set_index('Metric')
        
        # Draw a bar chart in Streamlit!
        st.bar_chart(importance_df)
        # Format the data exactly how the model expects it
        input_data = pd.DataFrame({
            'glucose_fasting': [glucose_input],
            'bmi': [bmi_input],
            'age': [age_input],
            'systolic_bp': [bp_input]
        })
        
        # Get the prediction (1 or 0)
        prediction = clf_model.predict(input_data)[0]
        
        # Bonus: Get the AI's confidence percentage!
        # predict_proba returns [Probability of 0, Probability of 1]
        probability = clf_model.predict_proba(input_data)[0][1] 
        
        # Display the results
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Diabetes detected.** (AI Confidence: {probability*100:.1f}%)")
        else:
            st.success(f"✅ **Low Risk of Diabetes.** (AI Confidence: {(1-probability)*100:.1f}%)")