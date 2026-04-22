import os
import logging
from pathlib import Path
from typing import Tuple
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------
# Custom Exception
# -----------------------------
class DataLoaderException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# -----------------------------
# Data Loader Function
# -----------------------------
def load_data(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and preprocess image dataset from directory.

    Args:
        data_dir (str): Path to dataset directory
        image_size (tuple): Image size (height, width)
        batch_size (int): Batch size
        validation_split (float): Fraction for validation
        seed (int): Random seed

    Returns:
        train_ds, val_ds (tf.data.Dataset)
    """

    try:
        logging.info("Starting data loading process...")

        data_path = Path(data_dir)

        if not data_path.exists():
            raise DataLoaderException(f"Data directory not found: {data_dir}")

        logging.info(f"Loading data from: {data_dir}")

        # -----------------------------
        # Training Dataset
        # -----------------------------
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )

        # -----------------------------
        # Validation Dataset
        # -----------------------------
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )

        logging.info("Dataset loaded successfully")

        # -----------------------------
        # Normalization (0–1 scaling)
        # -----------------------------
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        # -----------------------------
        # Performance Optimization
        # -----------------------------
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        logging.info("Data preprocessing completed")

        return train_ds, val_ds

    except Exception as e:
        logging.error(f"Error in data loading: {str(e)}")
        raise DataLoaderException(e)
# train_ds, val_ds = load_data(
#     data_dir="data/raw",
#     image_size=(224, 224),
#     batch_size=32
# )    
# print(f"Training dataset: {train_ds}")