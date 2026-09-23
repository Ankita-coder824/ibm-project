# =============================================================================
# Employee Remote Work Prediction – Streamlit Web Application
# =============================================================================
# Run with:   streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ── Page config (must be the very first Streamlit command) ───────────────────
st.set_page_config(
    page_title="Employee Remote Work Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join('data', 'employee_salary_dataset.csv')
MODEL_PATH = os.path.join('models', 'remote_work_model.pkl')
OUT_DIR    = 'outputs'

# ── Load model bundle (cached so it only loads once) ─────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

# ── Load dataset (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("💼 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home / Overview", "🔮 Predict Remote Work", "📊 Dashboard"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Employee Remote Work Prediction\n\n"
    "**Target:** remote_work\n\n"
    "**Type:** Multi-class Classification"
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – HOME / PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home / Overview":

    st.title("💼 Employee Remote Work Prediction")
    st.markdown("### Using Machine Learning to predict an employee's work arrangement")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
## 📌 Project Overview

This project uses **Machine Learning** to predict whether an employee is likely to:

- 🏠 **Work Remotely (Yes)**
- 🏢 **Work On-site (No)**
- 🔀 **Work in a Hybrid arrangement**

By analysing patterns in employee data such as job title, experience, education, 
industry, company size and salary, the model learns what kind of employees tend to 
have each type of work arrangement.

---

## 🎯 Project Objective

> **Given professional and employment information about an employee,
> predict their work arrangement (Remote / On-site / Hybrid).**

---

## ❓ Problem Statement

With the rise of remote work, understanding which factors influence remote work 
eligibility can help organisations plan better. This project builds a classification 
model that predicts the work arrangement of an employee based on features like:

- 🧑‍💼 Job Title
- 📅 Years of Experience
- 🎓 Education Level
- 🛠️ Number of Skills
- 🏭 Industry
- 🏢 Company Size
- 📍 Location
- 💰 Salary
- 📜 Certifications

---

## 🤖 How Machine Learning is Used

1. **Data Collection** – A real employee salary dataset with 2,50,000 records
2. **Preprocessing** – Missing values, encoding, scaling
3. **Model Training** – Multiple algorithms are trained and compared
4. **Evaluation** – Models are scored on Accuracy, Precision, Recall and F1 Score
5. **Prediction** – The best model predicts the work arrangement for new employees
        """)

    with col2:
        st.markdown("### 📂 Dataset Snapshot")
        try:
            df_preview = load_data().head(8)
            st.dataframe(df_preview, use_container_width=True)
        except Exception:
            st.warning("Dataset not found. Make sure `data/employee_salary_dataset.csv` exists.")

        bundle = load_model()
        if bundle:
            st.success(f"✅ Model loaded: **{bundle['model_name']}**")
            st.metric("Target Classes", len(bundle['target_classes']))
            st.metric("Input Features", len(bundle['feature_cols']))
        else:
            st.warning("⚠️ Model not found. Run `python train_model.py` first.")

    st.markdown("---")
    st.markdown("""
## 🧠 Machine Learning Algorithms Used

| Algorithm | Type | Notes |
|---|---|---|
| Logistic Regression | Linear | Fast, good baseline |
| Decision Tree | Tree-based | Interpretable |
| Random Forest | Ensemble | Best for tabular data |
| Gradient Boosting | Ensemble | High accuracy |

## 📏 Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Accuracy** | % of correct predictions overall |
| **Precision** | Of all predicted Remote, how many were actually Remote? |
| **Recall** | Of all actual Remote, how many were correctly found? |
| **F1 Score** | Balance between Precision and Recall |
| **Confusion Matrix** | Visual breakdown of correct vs wrong predictions |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Remote Work":

    st.title("🔮 Predict Remote Work Arrangement")
    st.markdown("Enter the employee details below and click **Predict Remote Work**.")
    st.markdown("---")

    bundle = load_model()
    if bundle is None:
        st.error("❌ Model file not found! Please run `python train_model.py` first.")
        st.stop()

    pipeline = bundle['pipeline']
    classes  = bundle['target_classes']

    # ── Load dataset to get unique values for dropdowns ───────────────────────
    try:
        df = load_data()
    except Exception:
        st.error("Dataset not found.")
        st.stop()

    # Helper: get sorted unique values
    def uvals(col):
        return sorted(df[col].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Job Information")
        job_title = st.selectbox("Job Title", uvals('job_title'))
        experience_years = st.slider("Experience Years", 0, int(df['experience_years'].max()), 5)
        education_level = st.selectbox(
            "Education Level",
            ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
        )

    with col2:
        st.subheader("🏢 Company Information")
        industry = st.selectbox("Industry", uvals('industry'))
        company_size = st.selectbox(
            "Company Size",
            ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
        )
        location = st.selectbox("Location", uvals('location'))

    with col3:
        st.subheader("📊 Skills & Salary")
        skills_count = st.slider(
            "Skills Count",
            int(df['skills_count'].min()),
            int(df['skills_count'].max()),
            5
        )
        certifications = st.slider(
            "Certifications",
            int(df['certifications'].min()),
            int(df['certifications'].max()),
            1
        )
        salary = st.number_input(
            "Salary (Annual)",
            min_value=int(df['salary'].min()),
            max_value=int(df['salary'].max()),
            value=int(df['salary'].median()),
            step=1000
        )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Remote Work", type="primary", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'job_title'        : job_title,
            'experience_years' : experience_years,
            'education_level'  : education_level,
            'skills_count'     : skills_count,
            'industry'         : industry,
            'company_size'     : company_size,
            'location'         : location,
            'certifications'   : certifications,
            'salary'           : salary,
        }])

        prediction = pipeline.predict(input_data)[0]

        # Map prediction to display text
        label_map = {
            'Yes'   : '🏠 Remote',
            'No'    : '🏢 On-site',
            'Hybrid': '🔀 Hybrid',
        }
        display_label = label_map.get(str(prediction), str(prediction))

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        if str(prediction) == 'Yes':
            st.success(f"### Predicted Work Arrangement: {display_label}")
        elif str(prediction) == 'No':
            st.error(f"### Predicted Work Arrangement: {display_label}")
        else:
            st.info(f"### Predicted Work Arrangement: {display_label}")

        # ── Probabilities ──────────────────────────────────────────────────────
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(input_data)[0]
            st.markdown("#### Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Work Arrangement': [label_map.get(c, c) for c in classes],
                'Probability (%)' : [round(p * 100, 1) for p in proba]
            })

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            with col_b:
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ['#2ecc71' if c == 'Yes' else '#e74c3c' if c == 'No'
                          else '#3498db' for c in classes]
                ax.barh(prob_df['Work Arrangement'], prob_df['Probability (%)'],
                        color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_title('Prediction Probability')
                ax.set_xlim(0, 100)
                for i, v in enumerate(prob_df['Probability (%)']):
                    ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ── Input summary ──────────────────────────────────────────────────────
        with st.expander("📄 Input Summary"):
            st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":

    st.title("📊 Project Dashboard")
    st.markdown("---")

    # Load data and model
    try:
        df = load_data()
    except Exception:
        st.error("Dataset not found.")
        st.stop()

    bundle = load_model()

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    total      = len(df)
    remote     = (df['remote_work'] == 'Yes').sum()
    on_site    = (df['remote_work'] == 'No').sum()
    hybrid     = (df['remote_work'] == 'Hybrid').sum()
    remote_pct = round(remote / total * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Employees", f"{total:,}")
    c2.metric("🏠 Remote",          f"{remote:,}", f"{remote_pct}%")
    c3.metric("🏢 On-site",         f"{on_site:,}")
    c4.metric("🔀 Hybrid",          f"{hybrid:,}")
    c5.metric("📊 Remote %",        f"{remote_pct}%")

    st.markdown("---")

    # ── Charts from outputs/ ──────────────────────────────────────────────────
    st.subheader("📈 Exploratory Data Analysis")
    chart_files = {
        "Target Distribution"         : "01_target_distribution.png",
        "Remote vs Education"         : "02_remote_vs_education.png",
        "Remote vs Company Size"      : "03_remote_vs_company_size.png",
        "Remote vs Industry"          : "04_remote_vs_industry.png",
        "Remote vs Location"          : "05_remote_vs_location.png",
        "Remote vs Job Title"         : "06_remote_vs_job_title.png",
        "Remote vs Experience"        : "07_remote_vs_experience.png",
        "Remote vs Salary"            : "08_remote_vs_salary.png",
        "Correlation Heatmap"         : "09_correlation_heatmap.png",
        "Skills Distribution"         : "10_skills_distribution.png",
    }

    for i, (title, fname) in enumerate(chart_files.items()):
        if i % 2 == 0:
            row = st.columns(2)
        fpath = os.path.join(OUT_DIR, fname)
        with row[i % 2]:
            if os.path.exists(fpath):
                st.markdown(f"**{title}**")
                st.image(fpath, use_container_width=True)
            else:
                st.info(f"Run train_model.py to generate: {fname}")

    st.markdown("---")

    # ── Model performance ─────────────────────────────────────────────────────
    st.subheader("🏆 Model Performance Comparison")
    if bundle:
        results_df = bundle['results_df']
        st.dataframe(
            results_df.style.highlight_max(axis=0, color='#d4edda').format("{:.2f}%"),
            use_container_width=True
        )
        st.success(f"✅ Best Model Selected: **{bundle['model_name']}**")

        mc_path = os.path.join(OUT_DIR, '11_model_comparison.png')
        cm_path = os.path.join(OUT_DIR, '12_confusion_matrix.png')
        fi_path = os.path.join(OUT_DIR, '13_feature_importance.png')

        cols = st.columns(2)
        if os.path.exists(mc_path):
            with cols[0]:
                st.markdown("**Model Comparison Chart**")
                st.image(mc_path, use_container_width=True)
        if os.path.exists(cm_path):
            with cols[1]:
                st.markdown(f"**Confusion Matrix – {bundle['model_name']}**")
                st.image(cm_path, use_container_width=True)

        if os.path.exists(fi_path):
            st.markdown("**Feature Importance (Random Forest)**")
            st.image(fi_path, use_container_width=True)
    else:
        st.warning("⚠️ Run `python train_model.py` to see model results.")

    st.markdown("---")

    # ── Dataset explorer ──────────────────────────────────────────────────────
    st.subheader("🔍 Dataset Explorer")
    show_n = st.slider("Number of rows to show", 5, 50, 10)
    filter_col = st.selectbox("Filter by Remote Work", ['All', 'Yes', 'No', 'Hybrid'])
    df_show = df if filter_col == 'All' else df[df['remote_work'] == filter_col]
    st.dataframe(df_show.head(show_n), use_container_width=True)
    st.caption(f"Showing {min(show_n, len(df_show))} of {len(df_show):,} rows")
