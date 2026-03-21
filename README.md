# Customer Churn Prediction Model

Predicts whether a customer will churn using **Logistic Regression** and **Random Forest**, with full EDA, feature engineering, hyperparameter tuning, and actionable business insights.

## Tech Stack
`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Joblib`

## Features
- Synthetic dataset generation (5,000 customers) with realistic churn dynamics
- Data cleaning — missing value imputation, label & one-hot encoding, scaling
- Feature engineering — charge-per-product, support-call rate, high-value flag
- Two models: Logistic Regression & Random Forest with 5-fold cross-validation
- GridSearchCV hyperparameter tuning on Random Forest
- **~85% accuracy** | ROC-AUC tracked for both models
- Visualisations: churn distribution, confusion matrices, ROC curves, feature importance
- Actionable retention insights printed to console

## Output Files (`outputs/churn/`)
| File | Description |
|------|-------------|
| `churn_distribution.png` | Class balance chart |
| `confusion_matrices.png` | Side-by-side CM for both models |
| `roc_curves.png` | ROC curves with AUC |
| `feature_importance.png` | Top 15 RF features |
| `monthly_charges_churn.png` | Charge distribution by churn |
| `best_churn_model.pkl` | Saved tuned Random Forest |
| `scaler.pkl` | Fitted StandardScaler |

## Quick Start
```bash
pip install -r requirements.txt
python churn_model.py
```

## Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~82% | ~0.88 |
| Random Forest | ~85% | ~0.92 |
