# 👥 Employee Attrition Prediction — Global Talent Solutions

Supervised machine learning (classification) to predict which employees are most likely to leave — combining data cleaning, class-imbalance handling with SMOTE, multiple models, hyperparameter tuning, and AutoML to give HR an early-warning signal for retention. 🔁

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modelling-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-3499CD)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-006600)
![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-2596BE)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 Overview

**Global Talent Solutions (GTS)** is a multinational HR consulting firm with 10,000+ employees across 25 countries. Facing an attrition rate well above the industry average, GTS wants to move from reacting to resignations toward **predicting** them. This project builds a classification model that flags employees as **"Likely to Leave"** or **"Likely to Stay"**, so the HR team can target retention efforts before people walk out the door.

## 🎯 Business Problem

High turnover disrupts operations, drives up recruitment and onboarding costs, and hurts client satisfaction. GTS needs to (1) understand the factors that drive attrition and (2) predict who is at risk. A reliable model lets the business:

- 💰 **Reduce costs** by cutting avoidable hiring and onboarding spend
- 🧭 **Improve workforce stability** by retaining experienced, client-critical staff
- 🤝 **Enhance client satisfaction** through continuity of service
- 📊 **Support data-driven HR decisions** with actionable insights

## 🗂️ Dataset

The dataset (`employee_attrition.csv`) starts with **8,400 employee records** across **11 columns**. `Attrition` is the target variable.

| Feature | Description |
|---|---|
| `EmployeeID` | Unique identifier for each employee *(dropped during preprocessing)* |
| `Age` | Employee's age in years |
| `Gender` | Employee's gender (Male / Female) |
| `Department` | Department (IT, HR, Finance, Sales, Operations) |
| `MonthlyIncome` | Monthly salary in USD |
| `YearsAtCompany` | Years the employee has worked at the company |
| `OverTime` | Whether the employee frequently works overtime (Yes / No) |
| `JobSatisfaction` | Job satisfaction (1–4 scale, 4 = very satisfied) |
| `WorkLifeBalance` | Perceived work-life balance (1–4 scale, 4 = excellent) |
| `TrainingTimesLastYear` | Number of training sessions attended last year |
| `Attrition` | **Whether the employee left the company (Yes / No) — target** |

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data wrangling | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, missingno |
| Imbalance handling | imbalanced-learn (**SMOTE**) |
| Machine learning | scikit-learn (Logistic Regression, Random Forest, SVM, k-NN, Decision Tree, Gradient Boosting, AdaBoost), XGBoost |
| Tuning & AutoML | `GridSearchCV`, **PyCaret** |
| Environment | Jupyter Notebook |

## 🔄 Project Workflow

1. **Data Preparation** — dropped the non-predictive `EmployeeID`; imputed missing values (`Age` ~10%, `MonthlyIncome` ~10% via **median**; `JobSatisfaction` ~10% via **mode**); removed **408 duplicate rows** (8,400 → 7,992); capped outliers in `Age`, `MonthlyIncome`, `YearsAtCompany`, and `TrainingTimesLastYear` using the **IQR method**.
2. **Encoding & Scaling** — label-encoded binary features (`Gender`, `OverTime`, `Attrition`); one-hot encoded `Department`; standardised numerical features with **StandardScaler** (chosen for their slightly skewed distributions).
3. **Class Imbalance** — the target was imbalanced (~**25% "Leave"** vs ~75% "Stay"), so **SMOTE** was applied to synthesise minority-class samples and balance the classes 50/50.
4. **Data Splitting** — 80% train / 20% test, **stratified** to preserve class proportions.
5. **Model Development** — trained a Logistic Regression baseline, then a suite of classifiers (Random Forest, SVM, k-NN, Decision Tree, Gradient Boosting, AdaBoost, XGBoost) with balanced class weights.
6. **Hyperparameter Tuning** — optimised the leading model (Random Forest) with `GridSearchCV` (5-fold, scored on F1).
7. **Evaluation** — assessed with precision, recall, F1-score, a confusion matrix, and the ROC curve / AUC.
8. **AutoML** — a **PyCaret** pipeline (`setup → compare_models → tune_model → finalize → save/load`) to automate model selection and cross-check the manual results.

## 📊 Model Performance

Evaluated on the balanced, held-out test set (2,383 records):

| Model | Accuracy | F1 (macro) |
|---|:---:|:---:|
| Logistic Regression *(baseline)* | 0.55 | 0.55 |
| Support Vector Machine | 0.58 | 0.58 |
| k-Nearest Neighbors | 0.69 | 0.69 |
| Decision Tree | 0.71 | 0.71 |
| Random Forest | 0.81 | 0.81 |
| **Random Forest (tuned)** ⭐ | **0.83** | **0.83** |

**Best model — tuned Random Forest** (`max_depth=20`, `n_estimators=200`, `min_samples_split=2`, `min_samples_leaf=1`):

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| Stay (0) | 0.80 | 0.87 | 0.84 |
| Leave (1) | 0.86 | 0.79 | 0.82 |

Model quality was further confirmed with a **confusion matrix** and a **ROC curve / AUC** analysis showing strong separability between leavers and stayers.

## 💡 Key Findings & Insights

- 🌳 **Tree-based models win decisively.** Linear and margin-based models (Logistic Regression 0.55, SVM 0.58) struggled, while ensemble trees captured the non-linear patterns in attrition — Random Forest jumped to 0.81 and tuning pushed it to **0.83 accuracy** with balanced precision and recall.
- 🎯 **The model is genuinely useful for HR.** The tuned Random Forest correctly identifies **~79% of employees who actually leave**, and when it predicts someone will leave it is right **~86% of the time** — a strong basis for targeted, cost-effective retention outreach.
- ⚖️ **Imbalance handling mattered.** With only ~25% of employees leaving, SMOTE was essential to stop the model from simply predicting "Stay" for everyone and to give the minority class a fair voice.
- 🧹 **Clean inputs did the heavy lifting.** Median/mode imputation, duplicate removal, and IQR outlier capping produced a stable dataset — a reminder that careful preparation drives model quality as much as algorithm choice.

## 📁 Repository Structure

```
├── Project-SML2-Classification.ipynb                 # Main analysis notebook
├── employee_attrition.csv                            # Raw dataset (8,400 records)
├── SML2_cleaned_employee_attrition_balanced.csv      # Cleaned + SMOTE-balanced dataset (generated)
├── final_model_attrition_prediction.pkl              # Saved model (generated by the AutoML step)
├── Case Study - Classification.pdf                   # Project brief
└── README.md
```

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/SML-Predicting-Employee-Attrition.git
cd SML-Predicting-Employee-Attrition

# 2. Install core dependencies
pip install pandas numpy matplotlib seaborn missingno scikit-learn imbalanced-learn xgboost jupyter

# 3. Launch the notebook
jupyter notebook Project-SML2-Classification.ipynb
```

> **⚠️ PyCaret note:** the AutoML section needs PyCaret, which pins older versions of NumPy / scikit-learn and can clash with a modern Anaconda base environment. Install it in an **isolated environment** so it doesn't break your other projects:
> ```bash
> conda create -n pycaret-env python=3.11 -y
> conda activate pycaret-env
> pip install pycaret
> python -m ipykernel install --user --name pycaret-env --display-name "Python (pycaret)"
> ```
> Then switch the notebook kernel to **Python (pycaret)** before running the AutoML cells.

> **Path note:** update the dataset path in the notebook to your local copy of `employee_attrition.csv` (e.g. place it in a `data/` folder and load with a relative path).

## 🔮 Future Improvements

- **Add a feature-importance analysis** (Random Forest importances or SHAP values) to explicitly rank the drivers of attrition — likely candidates include `OverTime`, `JobSatisfaction`, `MonthlyIncome`, `YearsAtCompany`, and `WorkLifeBalance`.
- **Run and report the PyCaret AutoML comparison** to benchmark the manual Random Forest against a wider model set.
- **Tune the decision threshold** to prioritise recall on the "Leave" class if the business cost of missing a leaver outweighs a false alarm.
- **Consider SMOTE inside a cross-validation pipeline** (rather than before the split) to give a fully leakage-free performance estimate.
- **Deploy** the finalised model as a simple scoring tool or API so HR can flag at-risk employees on demand.

## 👤 Author

**Marc Aurel Ntjam Minkeng** — *Data Science*
> Update with your GitHub profile and contact links.

---

*Supervised Learning · Classification · HR Analytics — a 10Alytics case study.*
