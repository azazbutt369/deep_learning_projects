# scripts/evaluate.py

import argparse

from src.data.dataset import prepare_dataset

from src.models.effnetv2b3 import build_effnetv2b3
from src.models.densenet121 import build_densenet121
from src.models.effnetb4 import build_effnetb4
from src.models.xception import build_xception

from src.training.trainer import compile_model

from src.evaluation.evaluator import evaluate_model, evaluate_ensemble


def main(args):
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Load data
    X_train, X_test, Y_train, Y_test = prepare_dataset(
        args.data_dir,
        labels,
        image_size=150
    )

    # Build models (same as notebook ensemble)
    models = [
        compile_model(build_effnetv2b3()),
        compile_model(build_densenet121()),
        compile_model(build_effnetb4()),
        compile_model(build_xception())
    ]

    # Evaluate each model
    for i, model in enumerate(models):
        print(f"\nEvaluating Model {i+1}")
        results = evaluate_model(model, X_test, Y_test)
        print(results)

    # Ensemble evaluation
    print("\nEvaluating Ensemble")
    ensemble_results = evaluate_ensemble(models, X_test, Y_test)
    print(ensemble_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)
