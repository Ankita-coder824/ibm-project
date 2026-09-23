# Employee Remote Work Prediction

## 📌 Project Title
**Employee Remote Work Prediction using Machine Learning**

---

## 🎯 Project Objective

Predict whether an employee works **Remotely**, **On-site**, or in a **Hybrid** arrangement based on their professional and employment information.

---

## ❓ Problem Statement

With the increasing adoption of remote work, organisations need to understand what factors drive different work arrangements. This project builds a Machine Learning classification model that predicts the work arrangement of an employee using features like job title, experience, education, industry, company size, salary, and certifications.

---

## 📂 Dataset Information

| Property | Value |
|---|---|
| File | `data/employee_salary_dataset.csv` |
| Total Records | 2,50,000 employees |
| Columns | 10 |

### Columns
| Column | Type | Description |
|---|---|---|
| job_title | Categorical | Role of the employee |
| experience_years | Numerical | Years of work experience |
| education_level | Categorical | Highest education (High School / Diploma / Bachelor / Master / PhD) |
| skills_count | Numerical | Number of skills the employee has |
| industry | Categorical | Industry the employee works in |
| company_size | Categorical | Size of the company (Startup / Small / Medium / Large / Enterprise) |
| location | Categorical | Country/city of work |
| remote_work | **Target** | Work arrangement: Yes / No / Hybrid |
| certifications | Numerical | Number of professional certifications |
| salary | Numerical | Annual salary |

---

## 🛠️ Features Used

**Input Features:**
- job_title, experience_years, education_level, skills_count, industry, company_size, location, certifications, salary

**Target Variable:**
- `remote_work` (Yes / No / Hybrid)

---

## 🔄 Data Preprocessing

1. **Drop duplicates** – Remove repeated rows
2. **Categorical Encoding** – OrdinalEncoder for categorical features
3. **Feature Scaling** – StandardScaler for numerical features
4. **ColumnTransformer** – Applies different transformations to numerical and categorical columns
5. **Pipeline** – Preprocessing + Model are combined into a single Pipeline to prevent data leakage

---

## 📊 Exploratory Data Analysis

Charts generated and saved to `outputs/`:

| File | Chart |
|---|---|
| 01_target_distribution.png | Distribution of remote_work values |
| 02_remote_vs_education.png | Remote work by education level |
| 03_remote_vs_company_size.png | Remote work by company size |
| 04_remote_vs_industry.png | Remote work by industry |
| 05_remote_vs_location.png | Remote work by location |
| 06_remote_vs_job_title.png | Remote work by job title |
| 07_remote_vs_experience.png | Salary distribution by remote work |
| 08_remote_vs_salary.png | Experience by remote work |
| 09_correlation_heatmap.png | Correlation of numerical features |
| 10_skills_distribution.png | Skills count by remote work |

---

## 🤖 Machine Learning Algorithms

| Model | Description |
|---|---|
| Logistic Regression | Simple linear classification model |
| Decision Tree | Tree-based model, easy to interpret |
| Random Forest | Ensemble of many decision trees |
| Gradient Boosting | Boosted ensemble with high accuracy |

---

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | % of correct predictions |
| **Precision** | Of predicted positives, how many are correct |
| **Recall** | Of actual positives, how many were found |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Visual table of correct vs incorrect predictions |

> For imbalanced classes, F1 Score is used as the primary selection metric.

---

## 🏆 Model Comparison

The best model is selected based on the **F1 Score** after training and evaluating all four models. Results are stored in the model bundle and displayed on the Dashboard.

---

## 💾 Final Selected Model

The model with the highest F1 Score is automatically selected and saved as:

```
models/remote_work_model.pkl
```

The saved file is a **Joblib bundle** containing:
- The trained sklearn Pipeline (Preprocessor + Classifier)
- Model name, target classes, feature columns
- All model results for the dashboard

---

## ⚙️ How to Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ How to Train the Model

```bash
python train_model.py
```

This will:
- Analyse the dataset
- Generate all EDA charts (saved to `outputs/`)
- Train all 4 models
- Evaluate and compare them
- Save the best model to `models/remote_work_model.pkl`

---

## 🚀 How to Run the Streamlit Application

```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

---

## 🔮 Example Prediction

**Input:**

| Feature | Value |
|---|---|
| Job Title | Data Analyst |
| Experience Years | 5 |
| Education Level | Bachelor |
| Skills Count | 10 |
| Industry | Technology |
| Company Size | Medium |
| Location | India |
| Certifications | 2 |
| Salary | 85000 |

**Output:**

```
Predicted Work Arrangement: Remote
```

---

## 🚀 Future Improvements

1. Hyperparameter tuning using GridSearchCV
2. Add XGBoost or LightGBM for better performance
3. Build a REST API using FastAPI for deployment
4. Deploy on Streamlit Cloud or Heroku
5. Add SHAP values for model explainability
6. Collect more recent real-world data

---

## 👨‍🎓 About

Built as a BCA final year project for Machine Learning.

> **Beginner-friendly**, clean code, with full comments and professional structure.
