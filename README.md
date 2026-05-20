# -customer_transaction_prediction

# Customer Transaction Prediction

Predicting whether a customer will make a transaction using machine learning — with a focus on handling severe class imbalance in banking data.

---

## Problem Statement

In banking, the majority of customers do not transact in a given period. This creates a heavily imbalanced dataset where a naive model simply predicts "no transaction" and still achieves high accuracy — but zero business value. The goal of this project is to build a model that reliably identifies the minority class (customers who *will* transact) to enable targeted outreach campaigns.

---

## Dataset

- **Source:** Provided by Rubixe AI Solutions as part of a structured analytics project program
- **Size:** 200,000+ banking records
- **Target:** Binary — whether a customer made a transaction (1) or not (0)
- **Class distribution:** ~9:1 imbalance (non-transaction majority)

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis across all features
- Correlation heatmap to identify multicollinearity
- Class imbalance visualization

### 2. Data Preprocessing
- Missing value treatment using median imputation
- IQR-based outlier detection and treatment
- Feature scaling where applicable

### 3. Model Comparison
Trained and evaluated 4 classification models:

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | Baseline | Linear decision boundary |
| Decision Tree | — | Prone to overfitting |
| Random Forest | — | Ensemble, handles noise well |
| **XGBoost** | **0.89** | Best performer |

### 4. Handling Class Imbalance
- Applied `scale_pos_weight` in XGBoost to penalize misclassification of minority class
- Tuned classification threshold to optimize F1-score on positive class
- Achieved **50% improvement in minority class detection** over baseline

### 5. Evaluation Metrics
- Primary: ROC-AUC (chosen due to class imbalance)
- Secondary: F1-score, Precision, Recall on positive class
- Confusion matrix analysis

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.89 |
| Minority class detection improvement | +50% over baseline |

---

## Business Impact

The final model produces a **customer scoring framework** — each customer gets a probability score of transacting. This allows the bank to:
- Rank and prioritize high-probability customers for outreach
- Reduce wasted campaign spend on unlikely converters
- Focus marketing budget where conversion likelihood is highest

---

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Repository Structure

```
customer-transaction-prediction/
│
└── customer_transaction_prediction.ipynb   # Full analysis and modeling notebook
```

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/Chethans246/customer-transaction-prediction.git
   cd customer-transaction-prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
   ```

3. Launch the notebook
   ```bash
   jupyter notebook customer_transaction_prediction.ipynb
   ```

---

## Key Learnings

- ROC-AUC is a far more meaningful metric than accuracy for imbalanced classification problems
- XGBoost's `scale_pos_weight` parameter is an effective and computationally cheap way to handle class imbalance without resampling
- Threshold tuning post-training can significantly improve minority class recall without retraining the model

---

## Author

**Chethan S** — Data Analyst  
[LinkedIn](https://linkedin.com/in/chethan-s-69b71b241) • [GitHub](https://github.com/Chethans246)
