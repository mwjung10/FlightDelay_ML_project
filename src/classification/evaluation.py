import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import matplotlib.pyplot as plt


def evaluate_model(model, X, y, cv: int = 5):
    """
    Perform stratified k-fold cross-validated evaluation (out-of-fold predictions).

    This function uses out-of-fold predictions to compute aggregated metrics so that the
    reported scores reflect cross-validated performance rather than a single train/test
    split. It requires that the estimator supports either `predict_proba` or
    `decision_function` for score-based metrics (ROC/PR). The function will fall back
    to decision_function if predict_proba is not available.

    Parameters:
        model: estimator (unfitted) implementing scikit-learn estimator API
        X: features (array-like or DataFrame)
        y: labels (array-like)
        cv: number of stratified folds (default: 5)

    Returns:
        metrics: dict with aggregated accuracy, classification_report, confusion_matrix,
                 roc_auc and pr_auc computed from out-of-fold predictions.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')

    y_scores = None
    try:
        y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
        y_scores = y_proba[:, 1]
    except Exception:
        try:
            y_scores = cross_val_predict(model, X, y, cv=skf, method='decision_function')
        except Exception:
            raise ValueError("Estimator must implement predict_proba or decision_function for score-based metrics")

    y_scores = np.asarray(y_scores).ravel()

    accuracy = accuracy_score(y, y_pred)
    class_report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_auc = auc(recall, precision)

    metrics = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
    }

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'ROC Curve (CV={cv})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', label=f'PR AUC = {pr_auc:.2f}')
    plt.title(f'Precision-Recall Curve (CV={cv})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

    return metrics


def compare_models(models, X_test, y_test):
    """
    Compare multiple classification models using evaluation metrics.

    Parameters:
        models: List of tuples (model_name, model).
        X_test: Test features.
        y_test: True labels for the test set.

    Returns:
        comparison: Dictionary containing evaluation metrics for each model.
    """
    comparison = {}

    for model_name, model in models:
        print(f"Evaluating model: {model_name}")
        metrics = evaluate_model(model, X_test, y_test)
        comparison[model_name] = metrics

    return comparison