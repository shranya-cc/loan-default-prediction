import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Dummy training data
df = pd.DataFrame({
    'Gender': ['Male', 'Female'],
    'Married': ['Yes', 'No'],
    'Dependents': ['0', '1'],
    'Education': ['Graduate', 'Not Graduate'],
    'CoapplicantIncome': [1000, 2000],
    'LoanAmount': [100, 200],
    'Credit_History': [1.0, 0.0],
    'Property_Area': ['Urban', 'Rural'],
    'Loan_Status': [1, 0]
})

# Splitting input and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Preprocessing
categorical = ['Gender', 'Married', 'Dependents', 'Education', 'Property_Area']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit pipeline
pipeline.fit(X, y)

# Save it in the same environment you're using for Streamlit
joblib.dump(pipeline, "loan_prediction_pipeline.pkl")
print("Model saved! Compatible with your current Python 3.13 and scikit-learn.")
