# scripts/train.py

import argparse

from src.data.dataset import prepare_dataset
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.effnetv2b3 import build_effnetv2b3
from src.models.densenet121 import build_densenet121
from src.models.effnetb4 import build_effnetb4
from src.models.xception import build_xception

from src.training.trainer import compile_model, train_model

def get_model(model_name, input_shape, num_classes):
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
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # dataset splitting
    X_train, X_test, Y_train, Y_test = prepare_dataset(
        args.data_dir,
        labels,
        image_size=150
    )

    # model initialisation
    model = get_model(
        args.model,
        input_shape=(150, 150, 3),
        num_classes=len(labels)
    )

    # model compilation
    model = compile_model(model)

    # model train
    model, history = train_model(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("Training completed.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="effnetv2b3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    main(args)
