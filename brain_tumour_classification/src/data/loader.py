import os
from typing import Tuple

def get_data_paths(base_dir: str) -> Tuple[str, str, str]:
    """
    Returns train, validation, and test directory paths.

    Expected structure:
    base_dir/
        train/
        val/
        test/
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    return train_dir, val_dir, test_dir
