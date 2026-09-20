"""
train.py
--------
Trains the defect-detection CNN on the casting product QC dataset,
evaluates it on the held-out test split, and exports:
  - models/defect_cnn.keras         (full Keras model)
  - models/defect_cnn.tflite        (quantized, edge-ready for Jetson)
  - reports/training_curves.png
  - reports/confusion_matrix.png
  - reports/metrics.json

Usage:
    python src/train.py --data_dir data_raw/casting_data/casting_data --epochs 12
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from model import build_defect_cnn, IMG_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def make_datasets(data_dir, img_size=IMG_SIZE, batch_size=32, val_split=0.15, seed=42):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        color_mode="grayscale",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        color_mode="grayscale",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        color_mode="grayscale",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names  # ['def_front', 'ok_front'] alphabetical

    # Light augmentation to improve generalization (edge devices see varied lighting/angles)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_raw/casting_data/casting_data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_model", default="models/defect_cnn.keras")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = make_datasets(
        args.data_dir, batch_size=args.batch_size
    )
    print("Class names (label 0 / 1):", class_names)

    model = build_defect_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_training_curves(history, os.path.join(args.reports_dir, "training_curves.png"))

    # ---- Evaluation on held-out test set ----
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print("Test metrics:", test_metrics)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).ravel()
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(args.reports_dir, "confusion_matrix.png"))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics_out = {
        "test_metrics": test_metrics,
        "classification_report": report,
        "class_names": class_names,
    }
    with open(os.path.join(args.reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ---- Save full model ----
    model.save(args.out_model)
    print(f"Saved Keras model to {args.out_model}")

    # ---- Export TFLite (quantized) for Jetson / edge inference ----
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = args.out_model.replace(".keras", ".tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {tflite_path}")


if __name__ == "__main__":
    main()
