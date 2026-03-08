# 💳 Credit Card Fraud Detection using Machine Learning

> Detecting fraudulent transactions using Logistic Regression, Random Forest & XGBoost on a real-world imbalanced dataset.

---

## 📸 Screenshots

### 🔍 Predict Transaction
![Predict](Screenshot%202026-03-08%20124310.png)

### 📊 Model Performance - Confusion Matrices
![Confusion Matrices](Screenshot%202026-03-08%20124500.png)

### 📊 Model Performance - Amount Distribution
![Amount Distribution](Screenshot%202026-03-08%20124518.png)

### ℹ️ About
![About](Screenshot%202026-03-08%20124542.png)

---

## 📌 Problem Statement
Credit card fraud causes billions in losses annually. This project builds an end-to-end ML pipeline to detect fraudulent transactions using the Kaggle Credit Card Fraud dataset — tackling one of the most challenging real-world ML problems: **extreme class imbalance**.

---

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud rate:** Only 0.17% (492 fraud cases) — highly imbalanced
- **Features:** V1–V28 (PCA-transformed for privacy), Amount, Time

---

## 🧠 ML Pipeline
```
Raw Data → EDA → Preprocessing → SMOTE → Model Training → Evaluation → Streamlit App
```

| Step | Details |
|------|---------|
| EDA | Class distribution, amount analysis, correlation heatmap |
| Preprocessing | StandardScaler on Amount & Time |
| Imbalance Handling | SMOTE (Synthetic Minority Oversampling Technique) |
| Models | Logistic Regression, Random Forest, XGBoost |
| Evaluation | Accuracy, ROC-AUC, Confusion Matrix, Classification Report |

---

## 🏆 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 97.43% | 0.97 |
| Random Forest | 99.94% | 0.99 |
| XGBoost | 99.92% | 0.9792 |

> **Best Model: XGBoost with ROC-AUC of 0.9792**

---

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit xgboost imbalanced-learn joblib pillow
python train.py
streamlit run app.py
```

---

## 📁 Project Structure
```
fraud-detection/
├── train.py                # ML pipeline script
├── fraud_detection.ipynb   # Jupyter notebook
├── app.py                  # Streamlit web app
└── README.md
```

---

## 🛠️ Tech Stack
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `SMOTE` · `Streamlit` · `Matplotlib` · `Seaborn`

---

## 👩‍💻 Author
**Nishita** | Computer Science Engineering, VIT Bhopal University  
GitHub: [github.com/nishitagajraj](https://github.com/nishitagajraj)
