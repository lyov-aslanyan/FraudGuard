
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TARGET = "Class"
TARGET_RECALL = 0.85

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
ERRORS_DIR = BASE_DIR / "reports" / "error_analysis"

for directory in (MODELS_DIR, FIGURES_DIR, ERRORS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND INSPECT DATA
# ============================================================

print("\n" + "=" * 60)
print("1. LOADING DATA")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that creditcard.csv is inside data/raw/."
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

if df[TARGET].isnull().any():
    raise ValueError("The target column contains missing values.")

print("\nOriginal class distribution:")
print(df[TARGET].value_counts())
print("\nOriginal class proportions:")
print(df[TARGET].value_counts(normalize=True))


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("2. CLEANING DATA")
print("=" * 60)

original_rows = len(df)

df = df.drop_duplicates().reset_index(drop=True)

removed_rows = original_rows - len(df)

print(f"Rows before cleaning: {original_rows}")
print(f"Duplicate rows removed: {removed_rows}")
print(f"Rows after cleaning: {len(df)}")

print("\nClass distribution after cleaning:")
print(df[TARGET].value_counts())
print("\nClass proportions after cleaning:")
print(df[TARGET].value_counts(normalize=True))

if df[TARGET].nunique() != 2:
    raise ValueError("This script expects a binary classification target.")


# ============================================================
# 4. SPLIT DATA: TRAIN / VALIDATION / TEST
# ============================================================

print("\n" + "=" * 60)
print("3. SPLITTING DATA")
print("=" * 60)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# 20% is reserved for the final test set.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

# 20% of the remaining 80% = 16% of the full dataset.
# Therefore, final proportions are 64% train, 16% validation,
# and 20% test.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    stratify=y_train_val,
    random_state=RANDOM_STATE,
)

print(f"Train:      {X_train.shape}, fraud cases: {int(y_train.sum())}")
print(f"Validation: {X_val.shape}, fraud cases: {int(y_val.sum())}")
print(f"Test:       {X_test.shape}, fraud cases: {int(y_test.sum())}")

print("\nSplit proportions:")
print(f"Train:      {len(X_train) / len(df):.1%}")
print(f"Validation: {len(X_val) / len(df):.1%}")
print(f"Test:       {len(X_test) / len(df):.1%}")


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def select_threshold(y_true, probabilities, target_recall=0.85):
    """
    Select the threshold with the highest precision among thresholds
    that achieve the required recall.

    Threshold selection uses validation data only.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    # precision and recall contain one more element than thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.where(recall >= target_recall)[0]

    if len(eligible) == 0:
        # If the target recall cannot be achieved, choose the
        # threshold with the highest recall.
        best_index = int(np.argmax(recall))
        reason = "Target recall could not be achieved."
    else:
        # Among eligible thresholds, maximize precision.
        best_index = eligible[
            np.argmax(precision[eligible])
        ]
        reason = "Target recall achieved."

    threshold = float(thresholds[best_index])

    return {
        "threshold": threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(
            f1_score(
                y_true,
                (probabilities >= threshold).astype(int),
                zero_division=0,
            )
        ),
        "reason": reason,
    }


def evaluate_model(name, y_true, probabilities, threshold):
    """Evaluate a model at a specified probability threshold."""
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )
    average_precision = average_precision_score(
        y_true, probabilities
    )
    roc_auc = roc_auc_score(y_true, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * 60)
    print(f"Threshold:       {threshold:.6f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-score:        {f1:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    return {
        "model": name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predictions": predictions,
    }


def save_confusion_matrix(y_true, predictions, title, output_path):
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 6. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("4. TRAINING LOGISTIC REGRESSION")
print("=" * 60)

lr_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

start_time = time.time()

lr_model.fit(X_train, y_train)

lr_training_time = time.time() - start_time

print(f"Training time: {lr_training_time:.2f} seconds")

lr_val_probabilities = lr_model.predict_proba(X_val)[:, 1]

lr_threshold_result = select_threshold(
    y_val,
    lr_val_probabilities,
    TARGET_RECALL,
)

lr_threshold = lr_threshold_result["threshold"]

print("\nLogistic Regression threshold selected on validation:")
print(f"Threshold: {lr_threshold:.6f}")
print(f"Precision: {lr_threshold_result['precision']:.4f}")
print(f"Recall: {lr_threshold_result['recall']:.4f}")
print(f"F1-score: {lr_threshold_result['f1']:.4f}")
print(lr_threshold_result["reason"])


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("5. TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

start_time = time.time()

rf_model.fit(X_train, y_train)

rf_training_time = time.time() - start_time

print(f"Training time: {rf_training_time:.2f} seconds")

rf_val_probabilities = rf_model.predict_proba(X_val)[:, 1]

rf_threshold_result = select_threshold(
    y_val,
    rf_val_probabilities,
    TARGET_RECALL,
)

best_rf_threshold = rf_threshold_result["threshold"]

print("\nRandom Forest threshold selected on validation:")
print(f"Threshold: {best_rf_threshold:.6f}")
print(f"Precision: {rf_threshold_result['precision']:.4f}")
print(f"Recall: {rf_threshold_result['recall']:.4f}")
print(f"F1-score: {rf_threshold_result['f1']:.4f}")
print(rf_threshold_result["reason"])


# ============================================================
# 8. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("6. FINAL TEST EVALUATION")
print("=" * 60)

# Important: thresholds were chosen on validation data.
# The test set is used only for final evaluation.

lr_test_probabilities = lr_model.predict_proba(X_test)[:, 1]
rf_test_probabilities = rf_model.predict_proba(X_test)[:, 1]

lr_results = evaluate_model(
    "Logistic Regression - Test",
    y_test,
    lr_test_probabilities,
    lr_threshold,
)

rf_results = evaluate_model(
    "Random Forest - Test",
    y_test,
    rf_test_probabilities,
    best_rf_threshold,
)


# ============================================================
# 9. COMPARE MODELS
# ============================================================

print("\n" + "=" * 60)
print("7. MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame(
    [
        {
            "Model": lr_results["model"],
            "Threshold": lr_results["threshold"],
            "Precision": lr_results["precision"],
            "Recall": lr_results["recall"],
            "F1": lr_results["f1"],
            "Average Precision": lr_results["average_precision"],
            "ROC-AUC": lr_results["roc_auc"],
            "False Positives": lr_results["fp"],
            "False Negatives": lr_results["fn"],
            "Training Time (s)": lr_training_time,
        },
        {
            "Model": rf_results["model"],
            "Threshold": rf_results["threshold"],
            "Precision": rf_results["precision"],
            "Recall": rf_results["recall"],
            "F1": rf_results["f1"],
            "Average Precision": rf_results["average_precision"],
            "ROC-AUC": rf_results["roc_auc"],
            "False Positives": rf_results["fp"],
            "False Negatives": rf_results["fn"],
            "Training Time (s)": rf_training_time,
        },
    ]
)

print(comparison.to_string(index=False))

comparison_path = BASE_DIR / "reports" / "model_comparison.csv"
comparison_path.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(comparison_path, index=False)

print(f"\nComparison saved to: {comparison_path}")


# ============================================================
# 10. PRECISION-RECALL CURVE
# ============================================================

print("\n" + "=" * 60)
print("8. SAVING PRECISION-RECALL CURVE")
print("=" * 60)

lr_precision, lr_recall, _ = precision_recall_curve(
    y_test,
    lr_test_probabilities,
)

rf_precision, rf_recall, _ = precision_recall_curve(
    y_test,
    rf_test_probabilities,
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    lr_recall,
    lr_precision,
    label=(
        "Logistic Regression "
        f"(AP={lr_results['average_precision']:.3f})"
    ),
)

ax.plot(
    rf_recall,
    rf_precision,
    label=(
        "Random Forest "
        f"(AP={rf_results['average_precision']:.3f})"
    ),
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - Test Set")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

pr_curve_path = FIGURES_DIR / "precision_recall_curve.png"
fig.savefig(pr_curve_path, dpi=150)
plt.close(fig)

print(f"Saved: {pr_curve_path}")


# ============================================================
# 11. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("9. SAVING CONFUSION MATRICES")
print("=" * 60)

save_confusion_matrix(
    y_test,
    lr_results["predictions"],
    "Logistic Regression - Test Set",
    FIGURES_DIR / "logistic_regression_confusion_matrix.png",
)

save_confusion_matrix(
    y_test,
    rf_results["predictions"],
    "Random Forest - Test Set",
    FIGURES_DIR / "random_forest_confusion_matrix.png",
)

print(f"Confusion matrices saved in: {FIGURES_DIR}")


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("10. FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": rf_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nTop 15 features:")
print(feature_importance.head(15).to_string(index=False))

feature_importance_path = (
    BASE_DIR / "reports" / "feature_importance.csv"
)
feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
feature_importance.to_csv(feature_importance_path, index=False)

top_features = feature_importance.head(15).sort_values(
    "importance", ascending=True
)

fig, ax = plt.subplots(figsize=(9, 7))

ax.barh(
    top_features["feature"],
    top_features["importance"],
)

ax.set_xlabel("Importance")
ax.set_title("Random Forest - Top 15 Feature Importances")

fig.tight_layout()

importance_plot_path = FIGURES_DIR / "feature_importance.png"
fig.savefig(importance_plot_path, dpi=150)
plt.close(fig)

print(f"Feature importance saved to: {feature_importance_path}")
print(f"Feature importance plot saved to: {importance_plot_path}")


# ============================================================
# 13. ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("11. ERROR ANALYSIS")
print("=" * 60)

rf_predictions = rf_results["predictions"]

# Store test features, actual labels, predicted labels,
# and predicted fraud probabilities.
test_analysis = X_test.copy()
test_analysis["actual"] = y_test
test_analysis["predicted"] = rf_predictions
test_analysis["fraud_probability"] = rf_test_probabilities

false_positives = test_analysis[
    (test_analysis["actual"] == 0)
    & (test_analysis["predicted"] == 1)
].copy()

false_negatives = test_analysis[
    (test_analysis["actual"] == 1)
    & (test_analysis["predicted"] == 0)
].copy()

fp_path = ERRORS_DIR / "false_positives.csv"
fn_path = ERRORS_DIR / "false_negatives.csv"

false_positives.to_csv(fp_path, index=False)
false_negatives.to_csv(fn_path, index=False)

print(f"False positives: {len(false_positives)}")
print(f"False negatives: {len(false_negatives)}")
print(f"False positives saved to: {fp_path}")
print(f"False negatives saved to: {fn_path}")


# ============================================================
# 14. SAVE FINAL MODEL AND THRESHOLD
# ============================================================

print("\n" + "=" * 60)
print("12. SAVING MODEL")
print("=" * 60)

model_path = MODELS_DIR / "fraud_model.joblib"
threshold_path = MODELS_DIR / "threshold.joblib"

joblib.dump(rf_model, model_path)
joblib.dump(best_rf_threshold, threshold_path)

print(f"Random Forest model saved to: {model_path}")
print(f"Selected threshold saved to: {threshold_path}")
print(f"Saved threshold value: {best_rf_threshold:.6f}")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT FINISHED")
print("=" * 60)

print(f"Dataset rows after cleaning: {len(df)}")
print(f"Target recall: {TARGET_RECALL:.2f}")
print(f"Random Forest threshold: {best_rf_threshold:.6f}")
print(f"Random Forest test precision: {rf_results['precision']:.4f}")
print(f"Random Forest test recall: {rf_results['recall']:.4f}")
print(f"Random Forest test F1: {rf_results['f1']:.4f}")
print(
    "Random Forest test Average Precision: "
    f"{rf_results['average_precision']:.4f}"
)
print(f"Random Forest test ROC-AUC: {rf_results['roc_auc']:.4f}")

print("\nAll tasks completed successfully.")
