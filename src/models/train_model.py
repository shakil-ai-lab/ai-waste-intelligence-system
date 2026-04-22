import logging
from pathlib import Path

import tensorflow as tf
import mlflow
import mlflow.tensorflow

from src.data.data_loader import load_data
from src.models.model import build_model, apply_fine_tuning


# -----------------------------
# Logging
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
# Training Pipeline
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

        # -----------------------------
        # Load Data
        # -----------------------------
        train_ds, val_ds = load_data(
            data_dir=str(data_dir),
            image_size=(224, 224),
            batch_size=32
        )

        # -----------------------------
        # MLflow Setup
        # -----------------------------
        mlflow.set_experiment("waste-classification")

        with mlflow.start_run():

            # =============================
            # 🔥 STAGE 1: Train Frozen Model
            # =============================
            logging.info("Stage 1: Training with frozen base model...")

            model = build_model(learning_rate=1e-3)

            history_stage1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10
            )

            # Save Stage 1 model
            stage1_path = model_dir / "stage1.keras"
            model.save(stage1_path)
            logging.info(f"Stage 1 model saved at: {stage1_path}")

            # Log Stage 1 metrics
            mlflow.log_metric("stage1_train_acc", history_stage1.history["accuracy"][-1])
            mlflow.log_metric("stage1_val_acc", history_stage1.history["val_accuracy"][-1])

            # =============================
            # 🔥 STAGE 2: Fine-Tuning
            # =============================
            logging.info("Stage 2: Fine-tuning model...")

            # Load SAME model (IMPORTANT)
            model = tf.keras.models.load_model(stage1_path)

            # Apply fine-tuning
            model = apply_fine_tuning(model, unfreeze_layers=10)

            history_stage2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10
            )

            # Save final model
            final_model_path = model_dir / "final_model.keras"
            model.save(final_model_path)
            logging.info(f"Final model saved at: {final_model_path}")

            # Log Stage 2 metrics
            mlflow.log_metric("stage2_train_acc", history_stage2.history["accuracy"][-1])
            mlflow.log_metric("stage2_val_acc", history_stage2.history["val_accuracy"][-1])

            # Log model in MLflow
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