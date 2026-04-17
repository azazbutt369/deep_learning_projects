import os
import cv2
from tqdm import tqdm


def load_images(data_dir, labels, image_size=150):
    """
    Loads images exactly as in notebook:
    - Reads from Training and Testing folders
    - Resizes using cv2
    - Stores in lists
    """

    X = []
    Y = []

    # Training data
    for label in labels:
        folder_path = os.path.join(data_dir, 'Training', label)
        for file in tqdm(os.listdir(folder_path), desc=f"Loading Training - {label}"):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            Y.append(label)

    # Testing data
    for label in labels:
        folder_path = os.path.join(data_dir, 'Testing', label)
        for file in tqdm(os.listdir(folder_path), desc=f"Loading Testing - {label}"):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            Y.append(label)

    return X, Y
