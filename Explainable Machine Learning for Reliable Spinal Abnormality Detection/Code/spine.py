import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Scikit-learn & Imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    ConfusionMatrixDisplay, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Statistics and Interpretability
import shap
from statsmodels.stats.contingency_tables import mcnemar

# Set seeds for reproducibility
np.random.seed(42)

# ==========================================
# 1. LOAD, CLEAN, AND RENAME DATA
# ==========================================
print("--- Loading and Cleaning Data ---")
df = pd.read_csv("Dataset_spine.csv")

# Clean column names and drop messy column
df.columns = df.columns.str.strip()
if "Unnamed: 13" in df.columns:
    df.drop(columns=["Unnamed: 13"], inplace=True)

# Medical Mapping
medical_names = {
    'Col1': 'Pelvic Incidence', 'Col2': 'Pelvic Tilt',
    'Col3': 'Lumbar Lordosis Angle', 'Col4': 'Sacral Slope',
    'Col5': 'Pelvic Radius', 'Col6': 'Degree Spondylolisthesis',
    'Col7': 'Pelvic Slope', 'Col8': 'Direct Tilt',
    'Col9': 'Thoracic Slope', 'Col10': 'Cervical Tilt',
    'Col11': 'Sacrum Angle', 'Col12': 'Scoliosis Slope'
}
df.rename(columns=medical_names, inplace=True)

# Encode Target: Abnormal = 1, Normal = 0
target_col = next((c for c in ['Class_att', 'Class', 'class', 'target'] if c in df.columns), None)
df['target'] = df[target_col].astype(str).str.strip().map({"Abnormal": 1, "Normal": 0})
df.drop(columns=[target_col], inplace=True)

print(f"Data Loaded. Target detected: '{target_col}'")
print(df['target'].value_counts())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
X = df.drop(columns=['target'])
y = df['target']

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Spinal Biomechanical Features")
plt.tight_layout()
plt.savefig("Figure1_Correlation_Heatmap.png", dpi=300)
plt.show()

# Boxplots for Key Features
key_features = ["Degree Spondylolisthesis", "Pelvic Incidence", "Lumbar Lordosis Angle"]
for feat in key_features:
    if feat in X.columns:
        plt.figure(figsize=(5, 4))
        sns.boxplot(x=y, y=X[feat])
        plt.xlabel("Class (0=Normal, 1=Abnormal)")
        plt.title(f"{feat} Distribution")
        plt.tight_layout()
        plt.savefig(f"Figure2_Boxplot_{feat.replace(' ', '_')}.png", dpi=300)
        plt.show()

# ==========================================
# 3. PREPROCESSING & SPLITTING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling (for LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE for Imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res_scaled, y_train_res_scaled = smote.fit_resample(X_train_scaled, y_train)

# ==========================================
# 4. MODEL TRAINING & EVALUATION
# ==========================================
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True), # True = Use Scaled Data
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42), False),
    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=42), False)
}

results_store = {}

print("\n--- Training and Evaluating Models ---")
for name, (model, use_scaled) in models.items():
    # Train
    train_x = X_train_res_scaled if use_scaled else X_train_res
    test_x = X_test_scaled if use_scaled else X_test
    
    model.fit(train_x, y_train_res)
    
    # Predict
    preds = model.predict(test_x)
    probs = model.predict_proba(test_x)[:, 1]
    
    results_store[name] = {"preds": preds, "probs": probs, "model": model}
    
    print(f"\n[{name}]")
    print(classification_report(y_test, preds))
    print(f"AUROC: {roc_auc_score(y_test, probs):.4f}")

# ==========================================
# 5. XGBOOST CALIBRATION & CONFUSION MATRIX
# ==========================================
best_name = "XGBoost"
best_model = results_store[best_name]["model"]
y_prob_best = results_store[best_name]["probs"]
y_pred_best = results_store[best_name]["preds"]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title(f"Confusion Matrix ({best_name})")
plt.savefig("Supp_Confusion_Matrix.png", dpi=300)
plt.show()

# Calibration Curve
frac_pos, mean_pred = calibration_curve(y_test, y_prob_best, n_bins=10)
plt.figure()
plt.plot(mean_pred, frac_pos, "s-", label=best_name)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.savefig("Supp_Calibration.png", dpi=300)
plt.show()

# ==========================================
# 6. SHAP INTERPRETABILITY (Random Forest)
# ==========================================
print("\n--- Generating SHAP Interpretability Plot ---")
rf_model = results_store["Random Forest"]["model"]
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Handle SHAP output dimensions
if isinstance(shap_values, list):
    shap_obj = shap_values[1] # Class 1 (Abnormal)
elif len(shap_values.shape) == 3:
    shap_obj = shap_values[:, :, 1]
else:
    shap_obj = shap_values

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_obj, X_test, show=False)
plt.title("Clinical Feature Importance (SHAP)")
plt.savefig("Figure3_SHAP_Summary.png", dpi=300)
plt.show()

# ==========================================
# 7. STATISTICAL VALIDATION
# ==========================================
# McNemar's Test: LR vs XGBoost
table = confusion_matrix(results_store["Logistic Regression"]["preds"] == y_test, 
                         results_store["XGBoost"]["preds"] == y_test)
mcnemar_result = mcnemar(table, exact=True)
print(f"\nMcNemar's Test (LR vs XGBoost) p-value: {mcnemar_result.pvalue:.4f}")

# Save Summary
summary_df = pd.DataFrame({
    "Model": results_store.keys(),
    "AUROC": [roc_auc_score(y_test, results_store[m]["probs"]) for m in results_store]
})
summary_df.to_csv("model_performance.csv", index=False)
print("✔ All processes complete. Results saved.")