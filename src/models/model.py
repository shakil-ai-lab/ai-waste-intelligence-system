import logging
from typing import Tuple

import tensorflow as tf
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
    learning_rate: float = 1e-3
) -> tf.keras.Model:

    try:
        logging.info("Building model...")

        # -----------------------------
        # Data Augmentation
        # -----------------------------
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="data_augmentation")

        # -----------------------------
        # Base Model (EfficientNet)
        # -----------------------------
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        base_model.trainable = False  # Stage 1: frozen

        # -----------------------------
        # Model Architecture
        # -----------------------------
        inputs = tf.keras.Input(shape=input_shape)

        x = data_augmentation(inputs)  # augmentation applied here
        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        # IMPORTANT: name base_model for easy access later
        model = models.Model(inputs, outputs, name="waste_classifier")

        # attach base_model as attribute (VERY IMPORTANT for fine-tuning)
        model.base_model = base_model

        # -----------------------------
        # Compile
        # -----------------------------
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logging.info("Model built successfully")

        return model

    except Exception as e:
        logging.error(f"Error in model building: {str(e)}")
        raise ModelBuilderException(e)


# -----------------------------
# Fine-Tuning Function
# -----------------------------
def apply_fine_tuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 10,
    learning_rate: float = 1e-5
) -> tf.keras.Model:

    try:
        logging.info("Applying fine-tuning...")

        # 🔥 FIND base model inside model layers
        base_model = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):  # EfficientNet is a Model
                base_model = layer
                break

        if base_model is None:
            raise ValueError("Base model (EfficientNet) not found!")

        # -----------------------------
        # Fine-Tuning
        # -----------------------------
        base_model.trainable = True

        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        # Recompile with LOW LR
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logging.info("Fine-tuning applied successfully")

        return model

    except Exception as e:
        logging.error(f"Error in fine-tuning: {str(e)}")
        raise ModelBuilderException(e)