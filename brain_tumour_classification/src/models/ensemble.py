# src/models/ensemble.py

import numpy as np

def average_ensemble(models, X):
    """
    This function performs simple averaging ensemble
    """

    predictions = []

    for model in models:
        preds = model.predict(X)
        predictions.append(preds)

    predictions = np.array(predictions)

    # averaging across models
    avg_preds = np.mean(predictions, axis=0)

    return avg_preds


def ensemble_predict(models, X):
    """
    Returns final class predictions
    """
    avg_preds = average_ensemble(models, X)
    return np.argmax(avg_preds, axis=1)
