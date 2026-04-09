"""
=============================================================
  AgriDrone AI — Model Training Script
  Dataset: Vegetable Disease Dataset (Kaggle)
  Model  : EfficientNetB4 with Transfer Learning
=============================================================

DATASET INSTRUCTIONS:
─────────────────────────────────────────────────────────────
DATASET:
   kaggle datasets download -d mgmitesh/plant-disease-detection-dataset

   38 classes, ~87,900 images (PlantVillage)
   Covers: Apple, Corn, Grape, Tomato, Potato, Pepper, etc.

FOLDER STRUCTURE (as provided by dataset):
   dataset/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Tomato___Late_blight/
   │   └── ... (38 classes)
   ├── valid/
   │   └── (same 38 subfolders)
   └── test/
       └── (same 38 subfolders)
─────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Config ──────────────────────────────────────────────────
IMG_SIZE   = 128
BATCH_SIZE = 64
EPOCHS     = 50
FINE_TUNE_EPOCHS = 20
LR         = 1e-3
FINE_LR    = 1e-5
DATA_DIR   = "dataset"
SAVE_DIR   = "model/saved_model"

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Data Generators ─────────────────────────────────────────
def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "valid"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    # Save class map for the Flask app
    class_map = {v: k for k, v in train_gen.class_indices.items()}
    with open(os.path.join(SAVE_DIR, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"[INFO] Classes detected: {train_gen.class_indices}")

    return train_gen, val_gen, test_gen


# ── Model Architecture ───────────────────────────────────────
def build_model(num_classes: int) -> Model:
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False  # Freeze for initial training

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base


# ── Callbacks ────────────────────────────────────────────────
def get_callbacks(phase: str):
    return [
        ModelCheckpoint(
            os.path.join(SAVE_DIR, f"best_model_{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=f"logs/{phase}", histogram_freq=1),
    ]


# ── Training ─────────────────────────────────────────────────
def train():
    print("\n" + "=" * 60)
    print("  AgriDrone AI — Training Pipeline")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_generators()
    num_classes = len(train_gen.class_indices)

    # ── Phase 1: Train top layers ────────────────────────────
    print("\n[PHASE 1] Training classification head (base frozen)...")
    model, base = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    model.summary()

    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=100,
        validation_steps=40,
        callbacks=get_callbacks("phase1"),
    )

    # ── Phase 2: Fine-tune top 60 layers of EfficientNet ────
    print("\n[PHASE 2] Fine-tuning top layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        steps_per_epoch=100,
        validation_steps=40,
        callbacks=get_callbacks("phase2"),
    )

    # ── Save final model ─────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, "best_model.keras")
    model.save(final_path)
    print(f"\n[SAVED] Final model → {final_path}")

    # ── Evaluate ─────────────────────────────────────────────
    print("\n[EVAL] Running evaluation on test set...")
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)

    # Save classification report
    with open(os.path.join(SAVE_DIR, "eval_report.txt"), "w") as f:
        f.write(report)

    # ── Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=16, pad=20)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
    print("[SAVED] confusion_matrix.png")

    # ── Training curves ──────────────────────────────────────
    _plot_history(h1, h2)
    print("\n[DONE] Training complete!")


def _plot_history(h1, h2):
    acc  = h1.history["accuracy"]       + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"]   + h2.history["val_accuracy"]
    loss = h1.history["loss"]           + h2.history["loss"]
    vloss= h1.history["val_loss"]       + h2.history["val_loss"]

    epochs_range = range(len(acc))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc,  label="Train Acc",  linewidth=2)
    axes[0].plot(epochs_range, vacc, label="Val Acc",    linewidth=2, linestyle="--")
    axes[0].axvline(len(h1.history["accuracy"]), color="r", linestyle=":", label="Fine-tune start")
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].legend()

    axes[1].plot(epochs_range, loss,  label="Train Loss", linewidth=2)
    axes[1].plot(epochs_range, vloss, label="Val Loss",   linewidth=2, linestyle="--")
    axes[1].axvline(len(h1.history["loss"]), color="r", linestyle=":", label="Fine-tune start")
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
    print("[SAVED] training_curves.png")


if __name__ == "__main__":
    train()
