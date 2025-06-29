## 💌 Loan Default Prediction using Machine Learning ❣️

This project predicts whether a loan applicant is likely to default based on various fictional and personal attributes.
It includes a two-part exploratory and modeling notebook as well as an user-friendly app for real-time predictions.

## Project Structure :- 

| File / Folder                                       | Description                                              |
|----------------------------------------------------|----------------------------------------------------------|
| `Loan_Default_Prediction_using_Machine_Learning.ipynb`     | **Part 1**: Exploratory data analysis and preprocessing       |
| `Loan_Default_Prediction_Using_Machine_Learning_Part_2.ipynb` | **Part 2**: Feature selection, model building, and evaluation |
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

. Data Cleaning and Imputation
. Exploratory Data Analysis (EDA)
. Feature Engineering and Encoding
. Model Building (Logistic Regression, Decision Tree, Random Forest)
. Evaluation (Accuracy, Confusion Matrix, Classification Report)


## Libraries and Tools Used :-

~NumPy, Pandas
~Seaborn, Matplotlib
~Scikit-Learn

~Jupyter Notebook

~Streamlit

---

## Limitations Found While Researching about the project!👇`(*>﹏<*)′

While substantial effort has been invested in cleaning, preprocessing, modeling, and deploying this project, analysis reveals that the dataset is biased. Key biases include:

* Gender imbalance (81% Male vs 19% Female)
* Disproportionate credit history availability (84% have credit history)
* Potential model favoring of certain demographics due to skewed training data

As a result, although the model achieves decent accuracy, it may not be fully fair or generalisable to real-world scenarios.

`A bias analysis notebook will be soon added to document and visualise these imbalances, along with suggestions to mitigate them in future iterations.


## Future Enhancements

* Hyperparameter tuning with GridSearchCV
* SHAP for model explainability
* Web deployment on Streamlit cloud
* Added input validation and improved UI styling

## Author ✨

Shranya Dutta
*Data Analyst in Progress| Machine Learning Enthusiast*

## License 
This project is licensed under the MIT License.
Feel free to use, modify, and share with proper attribution 
