# ============================================================
# Credit Card Fraud Detection - ML Pipeline
# Author: Nishita
# Dataset: Kaggle Credit Card Fraud Detection
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📥 Loading dataset...")
df = pd.read_csv(r"D:\OneDrive\Desktop\fraud_detection\creditcard.csv")
print(f"Shape: {df.shape}")
print(f"\nClass Distribution:\n{df['Class'].value_counts()}")
print(f"\nFraud %: {round(df['Class'].mean()*100, 4)}%")

# ─────────────────────────────────────────────
# 2. EDA - VISUALIZATIONS
# ─────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# Class imbalance plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['steelblue', 'crimson'])
plt.title('Class Distribution (0=Legit, 1=Fraud)')
plt.xticks([0, 1], ['Legitimate', 'Fraud'])
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("plots/class_distribution.png")
plt.close()
print("\n✅ Saved: plots/class_distribution.png")

# Transaction amount distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='steelblue', label='Legit')
plt.title('Legit Transaction Amounts')
plt.xlabel('Amount')

plt.subplot(1, 2, 2)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='crimson', label='Fraud')
plt.title('Fraud Transaction Amounts')
plt.xlabel('Amount')
plt.tight_layout()
plt.savefig("plots/amount_distribution.png")
plt.close()
print("✅ Saved: plots/amount_distribution.png")

# Correlation heatmap (top features)
plt.figure(figsize=(12, 8))
corr = df.corr()
top_features = corr['Class'].abs().sort_values(ascending=False).head(15).index
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap - Top Features')
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()
print("✅ Saved: plots/correlation_heatmap.png")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n🔧 Preprocessing...")

# Scale Amount and Time
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Handle class imbalance with SMOTE
print("\n⚖️  Applying SMOTE to handle class imbalance...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE - Train size: {X_train_sm.shape}")
print(f"Class distribution after SMOTE:\n{pd.Series(y_train_sm).value_counts()}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
print("\n🚀 Training models...\n")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "roc_auc": roc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    print(f"  ✅ Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS - RESULTS
# ─────────────────────────────────────────────

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues', xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_title(f'{name}\nAcc: {res["accuracy"]:.4f}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png")
plt.close()
print("\n✅ Saved: plots/confusion_matrices.png")

# ROC Curves
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curves.png")
plt.close()
print("✅ Saved: plots/roc_curves.png")

# Model comparison bar chart
plt.figure(figsize=(8, 5))
model_names = list(results.keys())
roc_scores = [results[m]['roc_auc'] for m in model_names]
colors = ['steelblue', 'seagreen', 'coral']
bars = plt.bar(model_names, roc_scores, color=colors, edgecolor='black')
plt.ylim(0.9, 1.0)
plt.ylabel('ROC-AUC Score')
plt.title('Model Comparison - ROC-AUC')
for bar, score in zip(bars, roc_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{score:.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("plots/model_comparison.png")
plt.close()
print("✅ Saved: plots/model_comparison.png")

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
top_feat = feat_imp.sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_feat.values, y=top_feat.index, palette='viridis')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()
print("✅ Saved: plots/feature_importance.png")

# ─────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────────
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")
print("✅ Saved: model/best_model.pkl")

print("\n🎉 Training complete! All plots saved in /plots folder.")
