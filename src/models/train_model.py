import os
import logging
from pathlib import Path

import tensorflow as tf
import mlflow
import mlflow.tensorflow

from src.data.data_loader import load_data
from src.models.model import build_model


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
class TrainingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# -----------------------------
# Training Function
# -----------------------------
def train():
    try:
        logging.info("Starting training pipeline...")

        # -----------------------------
        # Paths
        # -----------------------------
        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "data" / "raw"
        model_dir = base_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Data directory: {data_dir}")
        logging.info(f"Model directory: {model_dir}")

        # -----------------------------
        # Load Data
        # -----------------------------
        train_ds, val_ds = load_data(
            data_dir=str(data_dir),
            image_size=(224, 224),
            batch_size=32
        )

        # -----------------------------
        # Build Model
        # -----------------------------
        model = build_model(
            input_shape=(224, 224, 3),
            num_classes=6,
            learning_rate=0.001,
            fine_tune=False
        )

        model.summary()

        # -----------------------------
        # MLflow Setup
        # -----------------------------
        mlflow.set_experiment("waste-classification")

        with mlflow.start_run():

            # Log parameters
            mlflow.log_param("image_size", 224)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("model", "EfficientNetB0")

            # -----------------------------
            # Train Model
            # -----------------------------
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10
            )

            # -----------------------------
            # Log Metrics
            # -----------------------------
            final_train_acc = history.history["accuracy"][-1]
            final_val_acc = history.history["val_accuracy"][-1]

            mlflow.log_metric("train_accuracy", final_train_acc)
            mlflow.log_metric("val_accuracy", final_val_acc)

            # -----------------------------
            # Save Model
            # -----------------------------
            model_path = model_dir / "model.keras"
            model.save(model_path)

            logging.info(f"Model saved at: {model_path}")

            # Log model to MLflow
            mlflow.tensorflow.log_model(model, "model")

        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise TrainingException(e)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    train()
    print("Training pipeline executed successfully")