import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, accuracy_score, roc_curve, auc,
    precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def evaluate_model(model, X, y, cv: int = 5, no_plot: bool=False):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')

    y_scores = None
    try:
        y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
        y_scores = y_proba[:, 1]
    except Exception as e_proba:
        try:
            y_scores = cross_val_predict(model, X, y, cv=skf, method='decision_function')
        except Exception as e_decision:
            raise ValueError(
                f"Model must implement 'predict_proba' (Error: {e_proba}) or 'decision_function' (Error: {e_decision})"
            )

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
    if no_plot:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title(f'ROC Curve (CV={cv})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        prevalence = np.sum(y == 1) / len(y)
        plt.plot(recall, precision, color='green', label=f'PR AUC = {pr_auc:.2f}')
        plt.plot([0, 1], [prevalence, prevalence], color='gray', linestyle='--',
                label=f'Baseline (AUC = {prevalence:.3f})')
        plt.title(f'Precision-Recall Curve (CV={cv})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='best')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return metrics


def compare_models(models_dict, X, y, cv: int = 5, data_overrides=None):
    comparison_results = {}
    if data_overrides is None:
        data_overrides = {}

    print("Starting model evaluation (cross-validation)...")
    for model_name, model in models_dict.items():
        print(f"Evaluating: {model_name}...")

        if model_name in data_overrides:
            X_curr, y_curr = data_overrides[model_name]
            print(f"  -> Using override dataset ({len(X_curr)} samples)")
        else:
            X_curr, y_curr = X, y

        try:
            metrics = evaluate_model(model, X_curr, y_curr, cv=cv, no_plot=True)
            comparison_results[model_name] = metrics
            print(f"Completed: {model_name}")
        except Exception as e:
            print(f"FAILED: {model_name}. Error: {e}")
    print("Evaluation finished.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.50)')
    for model_name, metrics in comparison_results.items():
        ax1.plot(
            metrics['fpr'],
            metrics['tpr'],
            label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})"
        )
    ax1.set_title(f'ROC Curve Comparison (CV={cv})')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    prevalence = np.sum(y == 1) / len(y)
    ax2.plot([0, 1], [prevalence, prevalence], color='gray', linestyle='--',
             label=f'Baseline (AUC = {prevalence:.3f})')
    for model_name, metrics in comparison_results.items():
        ax2.plot(
            metrics['recall'],
            metrics['precision'],
            label=f"{model_name} (AUC = {metrics['pr_auc']:.3f})"
        )
    ax2.set_title(f'Precision-Recall Curve Comparison (CV={cv})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return comparison_results


def summarize_comparison(comparison_results, positive_class_label='1'):
    summary_data = []

    for model_name, metrics in comparison_results.items():
        report = metrics['classification_report']

        # Try to find the positive class key in various formats (str, int, float, bool)
        target_key = None
        candidates = [positive_class_label, 1, '1', 1.0, '1.0', True, 'True']

        for candidate in candidates:
            if candidate in report:
                target_key = candidate
                break

        if target_key is None:
            print(f"Warning: Key '{positive_class_label}' not found in "
                  f"report for {model_name}. Available keys: {list(report.keys())}. Using 'weighted avg' instead.")
            positive_metrics = report['weighted avg']
        else:
            positive_metrics = report[target_key]

        model_summary = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'ROC AUC': metrics['roc_auc'],
            'PR AUC': metrics['pr_auc'],
            'Precision (Class +)': positive_metrics['precision'],
            'Recall (Class +)': positive_metrics['recall'],
            'F1-score (Class +)': positive_metrics['f1-score'],
        }
        summary_data.append(model_summary)

    df_summary = pd.DataFrame(summary_data).set_index('Model')

    return df_summary.style.format("{:.4f}").background_gradient(cmap='viridis_r', axis=0)
