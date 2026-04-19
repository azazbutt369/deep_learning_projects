# scripts/predict.py

import argparse
import cv2
import numpy as np

from src.models.effnetv2b3 import build_effnetv2b3

def preprocess_image(image_path, image_size=150):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = np.expand_dims(img, axis=0)
    return img


def main(args):
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # load model (same structure as notebook)
    model = build_effnetv2b3()
  
    # preprocess image
    img = preprocess_image(args.image)

    # predict
    preds = model.predict(img)
    predicted_class = labels[np.argmax(preds)]

    print(f"Prediction: {predicted_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    main(args)
