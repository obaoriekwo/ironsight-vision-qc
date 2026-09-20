"""
inference.py
-------------
Simple CLI for running the trained model against a single image file —
useful for quick spot-checks without spinning up the API or camera loop.

Usage:
    python src/inference.py --image path/to/part.jpeg --model models/defect_cnn.keras
"""

import argparse

import numpy as np

from preprocess import load_and_preprocess

CLASS_NAMES = ["def_front", "ok_front"]


def predict_keras(model_path, image_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    arr = load_and_preprocess(image_path)
    batch = np.expand_dims(arr, axis=0)
    prob_ok = float(model.predict(batch, verbose=0)[0][0])
    return prob_ok


def predict_tflite(model_path, image_path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    arr = load_and_preprocess(image_path)
    batch = np.expand_dims(arr, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    prob_ok = float(interpreter.get_tensor(output_details[0]["index"])[0][0])
    return prob_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/defect_cnn.keras")
    args = parser.parse_args()

    if args.model.endswith(".tflite"):
        prob_ok = predict_tflite(args.model, args.image)
    else:
        prob_ok = predict_keras(args.model, args.image)

    is_ok = prob_ok >= 0.5
    label = "ok_front" if is_ok else "def_front"
    confidence = prob_ok if is_ok else 1 - prob_ok

    verdict = "PASS" if is_ok else "DEFECT"
    print(f"Image:      {args.image}")
    print(f"Verdict:    {verdict}")
    print(f"Label:      {label}")
    print(f"Confidence: {confidence*100:.2f}%")


if __name__ == "__main__":
    main()
