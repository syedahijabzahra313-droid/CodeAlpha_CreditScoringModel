## 📊 Dataset
- **Name:** Credit Card Fraud Detection
- **Source:** Kaggle
- **Rows:** 284,807 transactions
- **Features:** 30 (Time, V1-V28, Amount)
- **Target:** Class (0 = Normal, 1 = Fraud)
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

**1. Install libraries:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn