import streamlit as st
import joblib 
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')
file_path = 'saved_inputs.csv'

#Loading trained pipeline model
model = joblib.load("loan_prediction_pipeline.pkl")

st.title("Loan Approval Prediction App")

#Showing the model info
st.subheader('Model Info')
st.markdown('**Model Type:** Logistic Regression (GridSearchCV Tuned)')
accuracy = 0.80  # You can replace this with real test accuracy
st.markdown(f'**Test Accuracy:** `{accuracy * 100:.2f}%`')

st.markdown("---")

#Collecting the user inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
coapp_income = st.number_input('Coapplicant Income', min_value= 0.0)
loan_amount = st.number_input('Loan Amount', min_value = 0.0)
credit_history = st.selectbox('Credit History', ['Yes', 'No'])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

#Preparing the input DataFrame
input_df = pd.DataFrame([{
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'CoapplicantIncome': coapp_income,
    'LoanAmount': loan_amount,
    'Credit_History': 1.0 if credit_history == 'Yes' else 0.0,
    'Property_Area': property_area
}])

#Predicting and displaying the result
if st.button('Predict Loan Approval'):
    prediction = model.predict(input_df)[0]

    #Prediction with result
    if prediction == 1:
        st.success('Loan is likely to be Approved!')
        prediction_result = 'Approved'
    else:
        st.warning('Loan is likely to be Rejected.')
        prediction_result = 'Rejected'

    # Save inputs + prediction to CSV
    input_df['Prediction'] = prediction_result
    file_path = 'loan_predictions.csv'
    
    if os.path.exists(file_path):
        past_data = pd.read_csv(file_path)
        updated_data = pd.concat([past_data, input_df], ignore_index = True)
    else:
        updated_data = input_df
    
    updated_data.to_csv(file_path, index = False)
    st.info("Prediction saved to loan_predictions.csv")


