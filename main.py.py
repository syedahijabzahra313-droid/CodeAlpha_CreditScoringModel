# ============================================================
#   CodeAlpha Internship — Task 1: Credit Scoring Model
#   Dataset : Credit Card Fraud Detection (Kaggle)
#   File    : main.py
#   Folder  : CODEALPHA_CREDITSCORING/
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, average_precision_score
)

print("=" * 60)
print("   CodeAlpha — Task 1 : Credit Scoring Model")
print("   Dataset : Credit Card Fraud Detection")
print("=" * 60)

# ── 1. LOAD DATASET ───────────────────────────────────────────
print("\n[1/7] Loading dataset...")

df = pd.read_csv("dataset/creditcard.csv.csv")

print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")
print(f"   Normal transactions (0) : {(df['Class']==0).sum():,}")
print(f"   Fraud  transactions (1) : {(df['Class']==1).sum():,}")
print(f"   Fraud percentage        : {df['Class'].mean()*100:.3f}%")
print(f"   Missing values          : {df.isnull().sum().sum()}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n[2/7] Exploratory Data Analysis...")

os.makedirs("outputs", exist_ok=True)

# Class distribution
plt.figure(figsize=(6, 4))
counts = df['Class'].value_counts()
bars = plt.bar(['Normal (0)', 'Fraud (1)'],
               counts.values,
               color=['#2ecc71', '#e74c3c'],
               edgecolor='black', width=0.5)
for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1000,
             f'{count:,}', ha='center', fontweight='bold')
plt.title('Class Distribution\n(Normal vs Fraud)', fontsize=13)
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150)
plt.show()
print("   Saved → outputs/01_class_distribution.png")

# Amount distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
df[df['Class']==0]['Amount'].hist(
    bins=50, ax=axes[0], color='#2ecc71',
    edgecolor='black', alpha=0.8)
axes[0].set_title('Amount — Normal Transactions')
axes[0].set_xlabel('Amount ($)')
axes[0].set_ylabel('Count')

df[df['Class']==1]['Amount'].hist(
    bins=50, ax=axes[1], color='#e74c3c',
    edgecolor='black', alpha=0.8)
axes[1].set_title('Amount — Fraud Transactions')
axes[1].set_xlabel('Amount ($)')
plt.tight_layout()
plt.savefig("outputs/02_amount_distribution.png", dpi=150)
plt.show()
print("   Saved → outputs/02_amount_distribution.png")

# Correlation heatmap (Time, Amount + top V features)
selected = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5',
            'V7', 'V10', 'V12', 'V14', 'Amount', 'Class']
plt.figure(figsize=(12, 8))
sns.heatmap(df[selected].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/03_correlation_heatmap.png", dpi=150)
plt.show()
print("   Saved → outputs/03_correlation_heatmap.png")

# ── 3. PREPROCESSING ──────────────────────────────────────────
print("\n[3/7] Preprocessing...")

# Scale Time and Amount (V1-V28 already scaled by PCA)
scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_Time']   = scaler.fit_transform(df[['Time']])
df = df.drop(columns=['Amount', 'Time'])

X = df.drop('Class', axis=1)
y = df['Class']

# Using 50,000 normal + all 492 fraud
normal = df[df['Class']==0].sample(n=50000, random_state=42)
fraud  = df[df['Class']==1]
df_sample = pd.concat([normal, fraud]).sample(frac=1, random_state=42)

X = df_sample.drop('Class', axis=1)
y = df_sample['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train : {X_train.shape[0]:,} samples")
print(f"   Test  : {X_test.shape[0]:,} samples")
print(f"   Features : {X_train.shape[1]}")

# ── 4. MODEL TRAINING ─────────────────────────────────────────
print("\n[4/7] Training Models (please wait)...")

models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000,
                                random_state=42, class_weight='balanced'),
    "Decision Tree":        DecisionTreeClassifier(max_depth=6,
                                random_state=42, class_weight='balanced'),
    "Random Forest":        RandomForestClassifier(n_estimators=100,
                                random_state=42, class_weight='balanced',
                                n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100,
                                random_state=42, learning_rate=0.1),
}

results = {}
for name, model in models.items():
    print(f"   Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC":  roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
        "Avg-Prec": average_precision_score(y_test, y_prob),
    }
    print(f"✅  ROC-AUC = {results[name]['ROC-AUC']:.4f}")

# ── 5. BEST MODEL REPORT ──────────────────────────────────────
print("\n[5/7] Generating Reports...")

best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
best      = results[best_name]

print(f"\n   Best Model : {best_name}")
print(f"   ROC-AUC    : {best['ROC-AUC']:.4f}")
print(f"   F1-Score   : {best['F1-Score']:.4f}\n")
print(classification_report(y_test, best["y_pred"],
      target_names=["Normal", "Fraud"]))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"])
plt.title(f"Confusion Matrix — {best_name}", fontsize=13)
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/04_confusion_matrix.png", dpi=150)
plt.show()
print("   Saved → outputs/04_confusion_matrix.png")

# ── 6. ROC CURVES ─────────────────────────────────────────────
print("\n[6/7] Plotting Charts...")

plt.figure(figsize=(9, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, lw=2, color=color,
             label=f"{name} (AUC={res['ROC-AUC']:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — All Models", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/05_roc_curves.png", dpi=150)
plt.show()
print("   Saved → outputs/05_roc_curves.png")

# Precision-Recall Curve
plt.figure(figsize=(9, 6))
for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    plt.plot(rec, prec, lw=2, color=color,
             label=f"{name} (AP={res['Avg-Prec']:.3f})")
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curves — All Models", fontsize=14)
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/06_precision_recall.png", dpi=150)
plt.show()
print("   Saved → outputs/06_precision_recall.png")

# Model Comparison Bar Chart
metrics_df = pd.DataFrame({
    "Model":    list(results.keys()),
    "Accuracy": [r["Accuracy"] for r in results.values()],
    "ROC-AUC":  [r["ROC-AUC"]  for r in results.values()],
    "F1-Score": [r["F1-Score"] for r in results.values()],
}).set_index("Model")

metrics_df.plot(kind="bar", figsize=(11, 5), edgecolor="black",
                color=["#3498db", "#e74c3c", "#2ecc71"])
plt.title("Model Comparison", fontsize=14)
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.ylim(0.5, 1.05)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_model_comparison.png", dpi=150)
plt.show()
print("   Saved → outputs/07_model_comparison.png")

# Feature Importance
rf_model = results["Random Forest"]["model"]
feat_imp = pd.Series(rf_model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 5))
feat_imp[:15].plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Top 15 Feature Importances — Random Forest", fontsize=14)
plt.ylabel("Importance")
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/08_feature_importance.png", dpi=150)
plt.show()
print("   Saved → outputs/08_feature_importance.png")

# ── 7. FINAL SUMMARY ──────────────────────────────────────────
print("\n[7/7] Final Summary")
print("\n" + "=" * 65)
print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'F1-Score':>10}")
print("-" * 65)
for name, res in results.items():
    marker = " ← BEST" if name == best_name else ""
    print(f"{name:<25} {res['Accuracy']:>10.4f} "
          f"{res['ROC-AUC']:>10.4f} {res['F1-Score']:>10.4f}{marker}")
print("=" * 65)

print(f"""
✅ Task 1 Complete!
   Best Model  : {best_name}
   ROC-AUC     : {best['ROC-AUC']:.4f}
   F1-Score    : {best['F1-Score']:.4f}
   Charts saved in 'outputs/' folder (8 PNG files)
   GitHub repo : CodeAlpha_CreditScoringModel
""")