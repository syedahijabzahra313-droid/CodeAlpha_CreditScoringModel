# 💳 Credit Scoring Model — CodeAlpha Internship Task 1

## 📌 Objective
Predict fraudulent credit card transactions using machine learning classification algorithms.

## 📊 Dataset
- **Name:** Credit Card Fraud Detection (Kaggle)
- **Rows:** 284,807 transactions
- **Features:** Time, V1-V28, Amount
- **Target:** Class (0=Normal, 1=Fraud)
- **Fraud %:** 0.172%

## 🤖 Models Used
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Boosted ensemble model |

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Precision-Recall AUC

## ⚙️ How to Run
pip install numpy pandas matplotlib seaborn scikit-learn

python main.py

## 🗂️ Project Structure
CodeAlpha_CreditScoringModel/
├── dataset/
│   └── creditcard.csv.csv
├── outputs/
│   ├── 01_class_distribution.png
│   ├── 02_amount_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrix.png
│   ├── 05_roc_curves.png
│   ├── 06_precision_recall.png
│   ├── 07_model_comparison.png
│   └── 08_feature_importance.png
├── main.py
└── README.md

## 🛠️ Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 👤 Author
- **Name:** Hijab Zahra Naqvi
- **Internship:** CodeAlpha Machine Learning Intern
- **GitHub:** https://github.com/syedahijabzahra313-droid

## 🏢 Company
CodeAlpha — https://www.codealpha.tech
