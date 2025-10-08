import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model and return evaluation metrics.

    Parameters:
        model: Trained model with a predict_proba or predict method.
        X_test: Test features.
        y_test: True labels for the test set.

    Returns:
        metrics: Dictionary containing accuracy, classification report, confusion matrix, and AUC scores.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    metrics = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', label=f'PR AUC = {pr_auc:.2f}')
    plt.title('Precision-Recall Curve')
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