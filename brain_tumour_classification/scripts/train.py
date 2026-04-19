# scripts/train.py

import argparse

from src.data.dataset import prepare_dataset

from src.models.effnetv2b3 import build_effnetv2b3
from src.models.densenet121 import build_densenet121
from src.models.effnetb4 import build_effnetb4
from src.models.xception import build_xception

from src.training.trainer import compile_model, train_model

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger


def get_model(model_name, input_shape, num_classes):
    """
    Returns model based on config (strict mapping to notebook models)
    """
    if model_name == "effnetv2b3":
        return build_effnetv2b3(input_shape, num_classes)
    elif model_name == "densenet121":
        return build_densenet121(input_shape, num_classes)
    elif model_name == "effnetb4":
        return build_effnetb4(input_shape, num_classes)
    elif model_name == "xception":
        return build_xception(input_shape, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main(args):
    # --------------------------------------------------
    # Loading configuration
    # --------------------------------------------------
    config = load_config(args.config)

    logger = get_logger("train")

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------
    set_seed(
        seed=config["reproducibility"]["seed"],
        deterministic=config["reproducibility"]["deterministic"]
    )

    # --------------------------------------------------
    # Config extraction (STRICT notebook defaults)
    # --------------------------------------------------
    data_dir = args.data_dir if args.data_dir else config["data"]["data_dir"]
    labels = config["data"]["labels"]
    image_size = config["data"]["image_size"]

    input_shape = tuple(config["model"]["input_shape"])
    num_classes = config["model"]["num_classes"]

    epochs = args.epochs if args.epochs else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size else config["training"]["batch_size"]

    model_name = args.model if args.model else "effnetv2b3"

    logger.info(f"Using model: {model_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

    # --------------------------------------------------
    # Dataset (Notebook pipeline)
    # --------------------------------------------------
    X_train, X_test, Y_train, Y_test = prepare_dataset(
        data_dir,
        labels,
        image_size=image_size
    )

    logger.info("Dataset loaded successfully.")

    # --------------------------------------------------
    # Model creation
    # --------------------------------------------------
    model = get_model(
        model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes
    )

    logger.info("Model initialized.")

    # --------------------------------------------------
    # Compile (EXACT notebook config)
    # --------------------------------------------------
    model = compile_model(model)

    logger.info("Model compiled.")

    # --------------------------------------------------
    # Training (EXACT notebook behavior)
    # --------------------------------------------------
    model, history = train_model(
        model,
        X_train,
        Y_train,
        X_test,   # NOTE: matches notebook (used as validation)
        Y_test,
        epochs=epochs,
        batch_size=batch_size
    )

    logger.info("Training completed.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classification Model")

    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override dataset directory")

    parser.add_argument("--model", type=str, default=None,
                        choices=["effnetv2b3", "densenet121", "effnetb4", "xception"],
                        help="Model selection")

    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")

    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")

    args = parser.parse_args()

    main(args)
