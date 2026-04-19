# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_pred_labels(predictions):
    """
    Converts softmax outputs to class indices
    """
    return np.argmax(predictions, axis=1)

def get_true_labels(Y):
    """
    Converts one-hot encoded labels to class indices
    """
    return np.argmax(Y, axis=1)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
