# 💳 Credit Card Fraud Detection using Machine Learning

> Detecting fraudulent transactions using Logistic Regression, Random Forest & XGBoost on a real-world imbalanced dataset.

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

### Key Steps:
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
| Logistic Regression | ~97% | ~0.97 |
| Random Forest | ~99.9% | ~0.99 |
| XGBoost | ~99.9% | ~0.99 |

> **Note:** ROC-AUC is the primary metric due to class imbalance.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit xgboost imbalanced-learn joblib pillow
```

### 2. Download dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 3. Train the model
```bash
python train.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
fraud-detection/
├── creditcard.csv          # Dataset (download from Kaggle)
├── train.py                # ML pipeline script
├── fraud_detection.ipynb   # Jupyter notebook (step-by-step)
├── app.py                  # Streamlit web app
├── model/
│   ├── best_model.pkl      # Saved best model
│   └── feature_columns.pkl
├── plots/                  # Generated visualizations
└── README.md
```

---

## 🛠️ Tech Stack
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `Streamlit` · `Matplotlib` · `Seaborn`

---

## 👩‍💻 Author
**Nishita** | Computer Science Engineering, VIT Bhopal University  
GitHub: [github.com/nishitagajraj](https://github.com/nishitagajraj)