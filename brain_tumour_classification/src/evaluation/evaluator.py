# src/evaluation/evaluator.py

from .metrics import (
    get_pred_labels,
    get_true_labels,
    compute_accuracy,
    compute_classification_report,
    compute_confusion_matrix
)

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
