# src/evaluation/evaluator.py

from .metrics import (
    get_pred_labels,
    get_true_labels,
    compute_accuracy,
    compute_classification_report,
    compute_confusion_matrix
)

from src.models.ensemble import average_ensemble

def evaluate_model(model, X_test, Y_test):
    """
    EXACT evaluation flow from notebook
    """

    # predictions
    predictions = model.predict(X_test)

    # convert to labels
    y_pred = get_pred_labels(predictions)
    y_true = get_true_labels(Y_test)

    # evaluation metrics
    acc = compute_accuracy(y_true, y_pred)
    report = compute_classification_report(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }


def evaluate_ensemble(models, X_test, Y_test):
    """
    EXACT ensemble evaluation from notebook
    """

    predictions = average_ensemble(models, X_test)

    y_pred = get_pred_labels(predictions)
    y_true = get_true_labels(Y_test)

    acc = compute_accuracy(y_true, y_pred)
    report = compute_classification_report(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }
