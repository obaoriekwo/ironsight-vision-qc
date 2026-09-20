"""
model.py
--------
Defines the Convolutional Neural Network architecture used for
manufacturing surface-defect classification (defective vs. ok).

The architecture is intentionally lightweight (~1.2M params) so that
it can run at real-time frame rates on an NVIDIA Jetson (edge device)
after conversion to TensorFlow Lite / TensorRT.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = (128, 128)
CHANNELS = 1  # grayscale, matches source casting images


def build_defect_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS), l2=1e-4):
    """Builds and returns a compiled CNN for binary defect classification.

    Architecture: 4 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
    with increasing filter depth, followed by global average pooling and a
    small dense classifier head with dropout for regularization.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="defect_probability")(x)

    model = models.Model(inputs, outputs, name="defect_qc_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    m = build_defect_cnn()
    m.summary()
