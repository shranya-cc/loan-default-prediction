import streamlit as st
import joblib 
import pandas as pd
import warnings
import os
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
file_path = 'loan_predictions.csv'

#Loading the  model
model = joblib.load("loan_approval_model.pkl")

st.title("Loan Approval Prediction App")

#Model Info Section
st.subheader('Model Info')
accuracy = 0.618  #Updated accuracy from your training output
st.markdown('**Model Type:** XGBoost Classifier with SMOTE + GridSearchCV')
st.markdown(f'**Test Accuracy:** `{accuracy * 100:.2f}%`')

st.markdown("---")

#User Input Section
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
app_income = st.number_input('Applicant Income', min_value = 0.0, value = 0.0)
coapp_income = st.number_input('Coapplicant Income', min_value = 0.0, value = 0.0)
loan_amount = st.number_input('Loan Amount', min_value = 0.0, value = 0.0)
loan_term = st.number_input('Loan Amount Term (in days)', min_value = 1.0, value = 360.0)  #Changed the min_value to 1 to avoid zero
credit_history = st.selectbox('Credit History', ['Yes', 'No'])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

#Input validation
if loan_amount > (app_income + coapp_income) * 0.5:
    st.warning("Loan amount seems high compared to combined income. Please verify.")

if loan_term <= 0:
    st.warning("Loan Term must be positive and non-zero.")

#Prepare input DataFrame
input_df = pd.DataFrame([{
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': app_income,
    'CoapplicantIncome': coapp_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': 1.0 if credit_history == 'Yes' else 0.0,
    'Property_Area': property_area
}])

#Fix '3+' in Dependents and convert to float
input_df['Dependents'] = input_df['Dependents'].replace('3+', 3).astype(float)

#Feature Engineering
input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']

loan_term_months = input_df['Loan_Amount_Term'] / 30
monthly_interest_rate = 0.10 / 12

#Avoid divide by zero for EMI calculation
input_df['EMI'] = np.where(
    loan_term_months == 0,
    0,
    (input_df['LoanAmount'] * monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term_months) /
    ((1 + monthly_interest_rate) ** loan_term_months - 1)
)

input_df['DTI'] = input_df['EMI'] / (input_df['TotalIncome'] / 12 + 1e-6)  
#Show input preview
st.subheader("Input Preview (Before Prediction)")
st.dataframe(input_df)

#Helper function to get feature names from ColumnTransformer
def get_feature_names(column_transformer):
    feature_names = []

    for name, transformer, columns in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                names = list(transformer.get_feature_names_out(columns))
            else:
                names = list(columns)
            feature_names.extend(names)
    return feature_names

#Making the prediction
if st.button('Predict Loan Approval'):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        #Confidence messaging
        if proba >= 0.75:
            conf_msg = "High chance of approval! :D"
        elif proba >= 0.5:
            conf_msg = "Moderate chance of approval. :/"
        else:
            conf_msg = "Low chance of approval. :("

        if prediction == 1:
            st.success(f'Loan is likely to be Approved. {conf_msg}')
            prediction_result = 'Approved'
        else:
            st.warning(f'Loan is likely to be Rejected. {conf_msg}')
            prediction_result = 'Rejected'

        st.markdown(f"**Approval Confidence:** `{proba * 100:.2f}%`")

        #Feature Importance from XGBoost
        st.subheader("Feature Importance (XGBoost)")
        try:
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            feature_names = get_feature_names(preprocessor)

            importances = classifier.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by = 'Importance', ascending = False)

            st.dataframe(feat_imp)

        except Exception as e:
            st.error(f"Failed to get feature importance: {e}")

        #Add timestamp
        input_df['Prediction'] = prediction_result
        input_df['Timestamp'] = datetime.now()

        #Save to history file
        if os.path.exists(file_path):
            past_data = pd.read_csv(file_path)
            updated_data = pd.concat([past_data, input_df], ignore_index = True)
        else:
            updated_data = input_df

        updated_data.to_csv(file_path, index = False)
        st.info("Prediction saved to loan_predictions.csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")

#Prediction History Analysis
st.markdown("---")
st.subheader("Prediction History Summary")

if os.path.exists(file_path):
    pred_df = pd.read_csv(file_path)

    pred_counts = pred_df['Prediction'].value_counts()
    st.write("Prediction Count:")
    st.write(pred_counts)
    st.bar_chart(pred_counts)

    total = pred_counts.sum()
    for label in pred_counts.index:
        percent = (pred_counts[label] / total) * 100
        st.markdown(f"**{label}**: `{percent:.2f}%`")

    if 'Rejected' in pred_counts and pred_counts['Rejected'] / total > 0.75:
        st.error("Warning: Model may be biased toward Rejection.")
    elif 'Approved' in pred_counts and pred_counts['Approved'] / total > 0.75:
        st.warning("Model may be biased toward Approval.")
    else:
        st.success("Model prediction distribution looks balanced.")
else:
    st.info("No prediction history found yet.")
