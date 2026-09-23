# =============================================================================
# Employee Remote Work Prediction - Model Training Script
# =============================================================================
# This script:
#   1. Loads the dataset
#   2. Analyzes the data (EDA)
#   3. Preprocesses the data
#   4. Trains multiple classification models
#   5. Evaluates and compares all models
#   6. Saves the best pipeline for use in the Streamlit app
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join('data', 'employee_salary_dataset.csv')
MODEL_PATH  = os.path.join('models', 'remote_work_model.pkl')
OUTPUT_DIR  = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── Set style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='Set2')
COLORS = sns.color_palette('Set2')

# =============================================================================
# 1. LOAD DATASET
# =============================================================================
print("=" * 65)
print("  EMPLOYEE REMOTE WORK PREDICTION - Model Training")
print("=" * 65)

print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# 2. DATASET ANALYSIS
# =============================================================================
print("\n[2] Dataset Analysis")
print("-" * 45)
print(f"  Rows        : {df.shape[0]:,}")
print(f"  Columns     : {df.shape[1]}")
print(f"\n  Column Names & Data Types:")
print(df.dtypes.to_string())

print(f"\n  Missing Values per Column:")
missing = df.isnull().sum()
print(missing[missing >= 0].to_string())

print(f"\n  Duplicate Rows: {df.duplicated().sum():,}")

print(f"\n  Basic Statistical Summary:")
print(df.describe(include='all').to_string())

print(f"\n  Unique values of categorical columns:")
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print(f"    {col}: {df[col].unique().tolist()}")

print(f"\n  Target variable distribution (remote_work):")
print(df['remote_work'].value_counts().to_string())
print(df['remote_work'].value_counts(normalize=True).mul(100).round(2).astype(str).add('%').to_string())

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) – Save all charts to outputs/
# =============================================================================
print("\n[3] Generating EDA charts...")

# ── 3a. Target distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Remote Work Distribution', fontsize=14, fontweight='bold')

counts = df['remote_work'].value_counts()
axes[0].bar(counts.index, counts.values, color=COLORS[:len(counts)])
axes[0].set_title('Count')
axes[0].set_xlabel('Remote Work')
axes[0].set_ylabel('Number of Employees')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', fontsize=10)

axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=COLORS[:len(counts)], startangle=90)
axes[1].set_title('Percentage')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3b. Remote work vs Education Level ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
edu_order = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
edu_order = [e for e in edu_order if e in df['education_level'].unique()]
ct = pd.crosstab(df['education_level'], df['remote_work'])
ct = ct.reindex(edu_order)
ct.plot(kind='bar', ax=ax, color=COLORS[:ct.shape[1]])
ax.set_title('Remote Work vs Education Level', fontweight='bold')
ax.set_xlabel('Education Level')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_remote_vs_education.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3c. Remote work vs Company Size ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
size_order = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
size_order = [s for s in size_order if s in df['company_size'].unique()]
ct2 = pd.crosstab(df['company_size'], df['remote_work'])
ct2 = ct2.reindex(size_order)
ct2.plot(kind='bar', ax=ax, color=COLORS[:ct2.shape[1]])
ax.set_title('Remote Work vs Company Size', fontweight='bold')
ax.set_xlabel('Company Size')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_remote_vs_company_size.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3d. Remote work vs Industry ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ct3 = pd.crosstab(df['industry'], df['remote_work'])
ct3.plot(kind='bar', ax=ax, color=COLORS[:ct3.shape[1]])
ax.set_title('Remote Work vs Industry', fontweight='bold')
ax.set_xlabel('Industry')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_remote_vs_industry.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3e. Remote work vs Location ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ct4 = pd.crosstab(df['location'], df['remote_work'])
ct4.plot(kind='bar', ax=ax, color=COLORS[:ct4.shape[1]])
ax.set_title('Remote Work vs Location', fontweight='bold')
ax.set_xlabel('Location')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_remote_vs_location.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3f. Remote work vs Job Title ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ct5 = pd.crosstab(df['job_title'], df['remote_work'])
ct5.plot(kind='bar', ax=ax, color=COLORS[:ct5.shape[1]])
ax.set_title('Remote Work vs Job Title', fontweight='bold')
ax.set_xlabel('Job Title')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_remote_vs_job_title.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3g. Remote work vs Experience (boxplot) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='remote_work', y='experience_years', palette='Set2', ax=ax)
ax.set_title('Remote Work vs Experience Years', fontweight='bold')
ax.set_xlabel('Remote Work')
ax.set_ylabel('Experience Years')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_remote_vs_experience.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3h. Remote work vs Salary (boxplot) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='remote_work', y='salary', palette='Set2', ax=ax)
ax.set_title('Remote Work vs Salary', fontweight='bold')
ax.set_xlabel('Remote Work')
ax.set_ylabel('Salary')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_remote_vs_salary.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── 3i. Correlation heatmap (numerical only) ─────────────────────────────────
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if len(num_cols) >= 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap (Numerical Features)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_correlation_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close()

# ── 3j. Skills count distribution ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(data=df, x='skills_count', hue='remote_work', multiple='stack',
             palette='Set2', bins=20, ax=ax)
ax.set_title('Skills Count Distribution by Remote Work', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_skills_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()

print("    EDA charts saved to outputs/")

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================
print("\n[4] Preprocessing data...")

# Drop duplicates
df.drop_duplicates(inplace=True)
print(f"    After dropping duplicates: {df.shape[0]:,} rows")

# Define features and target
TARGET = 'remote_work'

# Feature columns – all except the target
FEATURE_COLS = [col for col in df.columns if col != TARGET]

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

# Encode target: Yes → 1, No → 0, Hybrid → 2
# We keep 3 classes: No / Yes / Hybrid
print(f"    Target classes: {sorted(y.unique())}")

# Identify numerical and categorical feature columns
NUM_FEATURES = ['experience_years', 'skills_count', 'certifications', 'salary']
CAT_FEATURES = ['job_title', 'education_level', 'industry', 'company_size', 'location']

# Make sure all expected columns actually exist
NUM_FEATURES = [c for c in NUM_FEATURES if c in X.columns]
CAT_FEATURES = [c for c in CAT_FEATURES if c in X.columns]

print(f"    Numerical features : {NUM_FEATURES}")
print(f"    Categorical features: {CAT_FEATURES}")

# ── ColumnTransformer ────────────────────────────────────────────────────────
# Numerical → StandardScaler
# Categorical → OrdinalEncoder (handles unseen values gracefully)
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, NUM_FEATURES),
    ('cat', categorical_transformer, CAT_FEATURES)
])

# ── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Training samples : {X_train.shape[0]:,}")
print(f"    Testing  samples : {X_test.shape[0]:,}")

# =============================================================================
# 5. TRAIN & EVALUATE MODELS
# =============================================================================
print("\n[5] Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
trained_pipelines = {}

for name, model in models.items():
    print(f"    -> Training {name}...")
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        'Accuracy' : round(acc  * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall'   : round(rec  * 100, 2),
        'F1 Score' : round(f1   * 100, 2),
    }
    trained_pipelines[name] = pipe

    print(f"       Accuracy={acc*100:.2f}%  Precision={prec*100:.2f}%  "
          f"Recall={rec*100:.2f}%  F1={f1*100:.2f}%")

# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).T
print("\n  Model Comparison:")
print(results_df.to_string())

# ── Model comparison chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(results_df))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=COLORS[i])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df.index, rotation=15)
ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_model_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Select best model by F1 Score ─────────────────────────────────────────────
best_model_name = results_df['F1 Score'].idxmax()
best_pipeline   = trained_pipelines[best_model_name]
print(f"\n  Best Model: {best_model_name} (F1={results_df.loc[best_model_name,'F1 Score']}%)")

# ── Confusion matrix for best model ──────────────────────────────────────────
y_pred_best = best_pipeline.predict(X_test)
classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred_best, labels=classes)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title(f'Confusion Matrix – {best_model_name}', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()

print(f"\n  Classification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best, zero_division=0))

# ── Feature importance (Random Forest or Gradient Boosting) ──────────────────
rf_pipe = trained_pipelines.get('Random Forest')
if rf_pipe:
    feat_names = NUM_FEATURES + CAT_FEATURES
    importances = rf_pipe.named_steps['classifier'].feature_importances_
    fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    fi_df.sort_values('Importance', ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='Set2', ax=ax)
    ax.set_title('Feature Importance – Random Forest', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '13_feature_importance.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Top features (Random Forest):")
    print(fi_df.to_string(index=False))

# =============================================================================
# 6. SAVE THE BEST MODEL PIPELINE
# =============================================================================
print(f"\n[6] Saving pipeline to {MODEL_PATH}...")

# Bundle everything the Streamlit app needs
model_bundle = {
    'pipeline'       : best_pipeline,
    'model_name'     : best_model_name,
    'target_classes' : classes,
    'feature_cols'   : FEATURE_COLS,
    'num_features'   : NUM_FEATURES,
    'cat_features'   : CAT_FEATURES,
    'results_df'     : results_df,
}
joblib.dump(model_bundle, MODEL_PATH)
print(f"    Pipeline saved successfully!")

# =============================================================================
# 7. QUICK PREDICTION TEST
# =============================================================================
print("\n[7] Quick prediction test...")
sample = pd.DataFrame([{
    'job_title'        : 'Data Analyst',
    'experience_years' : 5,
    'education_level'  : 'Bachelor',
    'skills_count'     : 10,
    'industry'         : 'Technology',
    'company_size'     : 'Medium',
    'location'         : 'India',
    'certifications'   : 2,
    'salary'           : 85000,
}])
pred  = best_pipeline.predict(sample)[0]
proba = best_pipeline.predict_proba(sample)[0] if hasattr(best_pipeline, 'predict_proba') else None
print(f"    Sample prediction : {pred}")
if proba is not None:
    for cls, p in zip(classes, proba):
        print(f"      {cls}: {p*100:.1f}%")

print("\n" + "=" * 65)
print("  Training complete! All outputs saved.")
print("=" * 65)
