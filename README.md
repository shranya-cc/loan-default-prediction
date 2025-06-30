## 💌 Loan Default Prediction using Machine Learning ❣️

This project predicts whether a loan applicant is likely to default based on various fictional and personal attributes.
It includes a two-part exploratory and modeling notebook as well as an user-friendly app for real-time predictions.

## Project Structure :- 

| File                                        | Description                                              |
|----------------------------------------------------|----------------------------------------------------------|
| `Loan_Default_Prediction_using_Machine_Learning.ipynb`     | **Part 1**: Exploratory data analysis and preprocessing       |
| `Loan_Default_Prediction_Using_Machine_Learning_Part_2.ipynb` | **Part 2**: Feature selection, model building, and evaluation |
|`Fairness_Evaluation_and_Bias_Mitigation_in_Loan_Default_Predicion`                    | Imbalance and Bias evaluation was done which were found in the dataset |
|`loan_prediction_app.py`        | Streamlit-based web application for loan predictions      |
|`loan_approval_model.pkl`                    | Trained model used in the app 
|`loan_predictions.csv`                    | No of predictions made by the app is saved in the format of CSV files

> The project is split across two notebooks for clarity and better observation
> **Part 1** focuses on data cleaning and EDA, while **Part 2** builds and evaluates ML models.

---

## Objectives 🌸

To assist financial institutions in identifying high-risk loan applicants by:

- Building an accurate classification model using historical loan data
- Creating a web-based app for real-time predictions
 
## Key Features 🧣

- Clean, modular analysis across two notebooks
- Future-proof pandas syntax (no chained assignments)
- Interactive prediction interface using Streamlit
- Visual insights from data distribution and correlations
- Multiple machine learning models compared for accuracy

## How to Run This App? 💫


1. **Clone the repository**

   * git clone https://github.com/shranya-cc/loan-default-prediction.git
   * cd loan-default-prediction
   
2. **Install required packages**
   pip install -r reuqirements.txt

3. **Run the app**

  streamlit run loan_prediction_app.py

---
## URL of the datasets for notebooks 1 and 2 😉

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

## ML Workflow 🎀

- Data Cleaning and Imputation
- Exploratory Data Analysis (EDA)
- Feature Engineering and Encoding
- Model Building (Logistic Regression, Decision Tree, Random Forest)
- Evaluation (Accuracy, Confusion Matrix, Classification Report)


## Libraries and Tools Used :-

~NumPy, Pandas
~Seaborn, Matplotlib
~Scikit-Learn

~Jupyter Notebook

~Streamlit

---

## Limitations Found While Researching about the project! (Data Imbalance and Model Bias) 🌷

The dataset used for this project exhibits a significant imbalance in the target variable distribution, with approximately 70% of loans labeled as "Rejected" and only about 30% as "Approved". This imbalance is common in loan approval datasets, as financial institutions tend to reject a majority of applications.

## Impact on Model Performance:

- The model tends to predict the majority class ("Rejected") more often, leading to a skewed prediction distribution reflecting the dataset bias.

- Despite extensive efforts — including applying techniques such as SMOTE (Synthetic Minority Over-sampling Technique), hyperparameter tuning, different resampling strategies, and cost-sensitive learning — the imbalance remained challenging to fully overcome.

- These efforts did not significantly improve the model's ability to accurately identify the minority class ("Approved"), resulting in lower sensitivity and limiting practical usefulness.

- Accuracy alone is not a sufficient metric in this context, so evaluation metrics such as precision, recall, and F1-score were considered to better understand model performance.

## Additional Biases:

- Gender imbalance (81% Male vs 19% Female) and credit history availability (84% have credit history) may introduce unintended demographic biases in predictions.

- These imbalances suggest the model may favour certain groups over others, which is a critical consideration for fairness and ethical AI.

## Bias Analysis Note 😉 

A dedicated notebook has been added to perform detailed bias analysis on the dataset, including visualization of imbalances and mitigation strategies.

---

## Author ✨

Shranya Dutta
*Data Analyst in Progress| Machine Learning and Data Analysis Enthusiast*

## License 🧣
This project is licensed under the MIT License.
Feel free to use, modify, and share with proper attribution 
