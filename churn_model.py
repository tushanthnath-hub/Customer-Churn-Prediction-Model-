
# Customer Churn Prediction Model
# Tools: Python, Scikit-learn, Pandas, Matplotlib, Seaborn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import joblib
import os


# 1. Generate Synthetic Dataset


def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic customer churn dataset."""
    rng = np.random.default_rng(random_state)

    tenure          = rng.integers(1, 72, n_samples)
    monthly_charges = rng.uniform(20, 120, n_samples).round(2)
    total_charges   = (monthly_charges * tenure + rng.normal(0, 50, n_samples)).clip(0).round(2)
    num_products    = rng.integers(1, 5, n_samples)
    support_calls   = rng.integers(0, 10, n_samples)
    age             = rng.integers(18, 70, n_samples)

    contract_type   = rng.choice(["Month-to-Month", "One Year", "Two Year"], n_samples,
                                  p=[0.55, 0.25, 0.20])
    internet_svc    = rng.choice(["DSL", "Fiber Optic", "No"], n_samples,
                                  p=[0.35, 0.45, 0.20])
    payment_method  = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        n_samples, p=[0.35, 0.25, 0.20, 0.20]
    )
    gender          = rng.choice(["Male", "Female"], n_samples)
    paperless_bill  = rng.choice(["Yes", "No"], n_samples)
    senior_citizen  = rng.integers(0, 2, n_samples)

    # Churn probability influenced by real-world factors
    churn_prob = (
        0.35 * (contract_type == "Month-to-Month").astype(float)
        + 0.20 * (internet_svc == "Fiber Optic").astype(float)
        + 0.15 * (support_calls > 5).astype(float)
        + 0.10 * (monthly_charges > 80).astype(float)
        - 0.20 * (tenure > 36).astype(float)
        - 0.10 * (num_products > 2).astype(float)
        + rng.uniform(0, 0.1, n_samples)
    ).clip(0.05, 0.95)

    churn = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "CustomerID":      [f"CUST{str(i).zfill(5)}" for i in range(n_samples)],
        "Age":             age,
        "Gender":          gender,
        "SeniorCitizen":   senior_citizen,
        "Tenure":          tenure,
        "NumProducts":     num_products,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
        "SupportCalls":    support_calls,
        "ContractType":    contract_type,
        "InternetService": internet_svc,
        "PaymentMethod":   payment_method,
        "PaperlessBilling": paperless_bill,
        "Churn":           churn,
    })

    # Inject ~3% missing values in TotalCharges
    missing_idx = rng.choice(df.index, size=int(0.03 * n_samples), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    return df

# 2. Data Preprocessing

def preprocess(df: pd.DataFrame):
    """Clean, encode, and scale features."""
    df = df.copy()

    # Drop ID (not a feature)
    df.drop(columns=["CustomerID"], inplace=True)

    # Handle missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode binary categoricals
    binary_cols = ["Gender", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode multi-class categoricals
    df = pd.get_dummies(df, columns=["ContractType", "InternetService", "PaymentMethod"],
                        drop_first=True)

    # Feature / target split
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Scale numerical features
    num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls"]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, scaler


# 3. Feature Engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgMonthlyCharge"]   = df["TotalCharges"] / (df["Tenure"] + 1)
    df["ChargePerProduct"]   = df["MonthlyCharges"] / (df["NumProducts"] + 1)
    df["SupportCallRate"]    = df["SupportCalls"] / (df["Tenure"] + 1)
    df["HighValueCustomer"]  = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    return df


# 4. Model Training & Evaluation


def train_evaluate(X_train, X_test, y_train, y_test, output_dir: str):
    results = {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]
        acc         = accuracy_score(y_test, y_pred)
        roc         = roc_auc_score(y_test, y_prob)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model":      model,
            "accuracy":   acc,
            "roc_auc":    roc,
            "cv_mean":    cv_scores.mean(),
            "cv_std":     cv_scores.std(),
            "y_pred":     y_pred,
            "y_prob":     y_prob,
        }

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  ROC-AUC   : {roc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}")

    return results

# 5. Hyperparameter Tuning (Random Forest)


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"\n[Tuning] Best RF params : {grid.best_params_}")
    print(f"[Tuning] Best ROC-AUC   : {grid.best_score_:.4f}")
    return grid.best_estimator_



# 6. Visualisations

def plot_all(df_raw, results, X_test, y_test, feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # --- 6a. Churn distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_raw["Churn"].value_counts()
    ax.bar(["Retained", "Churned"], counts.values, color=["steelblue", "tomato"], edgecolor="white")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v}\n({v/len(df_raw)*100:.1f}%)", ha="center", fontsize=10)
    ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_distribution.png", dpi=150)
    plt.close()

    # --- 6b. Confusion matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.2%}", fontsize=12)
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150)
    plt.close()

    # --- 6c. ROC curves ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "tomato"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=150)
    plt.close()

    # --- 6d. Feature importance (Random Forest) ---
    rf_model = results["Random Forest"]["model"]
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": rf_model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax, palette="Blues_r")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close()

    # --- 6e. Monthly charges vs Churn ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in zip([0, 1], ["steelblue", "tomato"]):
        ax.hist(df_raw[df_raw["Churn"] == label]["MonthlyCharges"],
                bins=30, alpha=0.6, color=color,
                label="Retained" if label == 0 else "Churned")
    ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Count")
    ax.set_title("Monthly Charges by Churn Status", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_charges_churn.png", dpi=150)
    plt.close()

    print(f"\n[Plots] Saved to '{output_dir}/'")


# ─────────────────────────────────────────────
# 7. Actionable Insights
# ─────────────────────────────────────────────

def actionable_insights(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  ACTIONABLE INSIGHTS FOR CUSTOMER RETENTION")
    print("="*55)

    churn_rate = df["Churn"].mean() * 100
    print(f"\n  Overall Churn Rate : {churn_rate:.2f}%")

    mtm_churn = df[df["ContractType"] == "Month-to-Month"]["Churn"].mean() * 100
    print(f"\n  1. Month-to-Month churn rate ({mtm_churn:.1f}%) is highest.")
    print("     → Offer discounts to switch to annual contracts.")

    high_support = df[df["SupportCalls"] > 5]["Churn"].mean() * 100
    print(f"\n  2. Customers with >5 support calls churn at {high_support:.1f}%.")
    print("     → Proactively reach out after 3rd support call.")

    high_charge_churn = df[df["MonthlyCharges"] > 80]["Churn"].mean() * 100
    print(f"\n  3. High monthly charge (>$80) churn rate: {high_charge_churn:.1f}%.")
    print("     → Introduce loyalty rewards for high-paying customers.")

    long_tenure = df[df["Tenure"] > 36]["Churn"].mean() * 100
    print(f"\n  4. Long-tenure customers (>3 yrs) churn at only {long_tenure:.1f}%.")
    print("     → Focus retention efforts on customers in first 6 months.")
    print()



# 8. Main Pipeline
def main():
    OUTPUT_DIR = "outputs/churn"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/6] Generating dataset …")
    df_raw = generate_dataset(n_samples=5000)
    print(f"      Shape: {df_raw.shape} | Churn rate: {df_raw['Churn'].mean():.2%}")

    print("\n[2/6] Feature engineering …")
    df_fe = engineer_features(df_raw)

    print("\n[3/6] Preprocessing …")
    X, y, scaler = preprocess(df_fe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    print("\n[4/6] Training & evaluating models …")
    results = train_evaluate(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    print("\n[5/6] Hyperparameter tuning (Random Forest) …")
    best_rf = tune_random_forest(X_train, y_train)
    y_pred_tuned = best_rf.predict(X_test)
    print(f"      Tuned RF Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

    print("\n[6/6] Generating visualisations …")
    plot_all(df_raw, results, X_test, y_test, X.columns.tolist(), OUTPUT_DIR)

    # Save best model
    joblib.dump(best_rf, f"{OUTPUT_DIR}/best_churn_model.pkl")
    joblib.dump(scaler,  f"{OUTPUT_DIR}/scaler.pkl")
    print(f"[Model] Saved to '{OUTPUT_DIR}/'")

    actionable_insights(df_raw)


if __name__ == "__main__":
    main()
