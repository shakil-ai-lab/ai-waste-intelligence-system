import os
import logging
from typing import Tuple

import tensorflow as tf
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras import layers, models


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
class ModelBuilderException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# -----------------------------
# Model Builder Function
# -----------------------------
def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    learning_rate: float = 0.001,
    fine_tune: bool = False
) -> tf.keras.Model:
    """
    Build EfficientNet-based classification model

    Args:
        input_shape (tuple): Input image shape
        num_classes (int): Number of output classes
        learning_rate (float): Learning rate
        fine_tune (bool): Whether to unfreeze top layers

    Returns:
        model (tf.keras.Model)
    """

    try:
        logging.info("Starting model building...")

        # -----------------------------
        # Load Pretrained EfficientNet
        # -----------------------------
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        logging.info("EfficientNetB0 loaded with ImageNet weights")

        # Freeze base model
        base_model.trainable = False

        # -----------------------------
        # Custom Classification Head
        # -----------------------------
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=outputs)

        # -----------------------------
        # Fine-Tuning (Optional)
        # -----------------------------
        if fine_tune:
            logging.info("Applying fine-tuning...")

            # Unfreeze top layers only
            for layer in base_model.layers[-20:]:
                layer.trainable = True

        # -----------------------------
        # Compile Model
        # -----------------------------
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logging.info("Model compiled successfully")

        return model

    except Exception as e:
        logging.error(f"Error in model building: {str(e)}")
        raise ModelBuilderException(e)