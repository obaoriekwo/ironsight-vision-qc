"""
run_camera.py
--------------
Real-time edge inference loop, intended to run on an NVIDIA Jetson
(Nano / Xavier NX / Orin) connected to a line-side USB or CSI camera.

Loads the quantized TFLite model (fast on Jetson's CPU/GPU via the
TFLite runtime, and a drop-in candidate for TensorRT conversion) and
classifies each captured frame as OK or DEFECT in real time, drawing
a HUD overlay for the operator and optionally pushing results to the
backend API (see app/backend/app.py) for logging / the dashboard.

Usage:
    python src/run_camera.py --camera 0 --model models/defect_cnn.tflite
    python src/run_camera.py --camera 0 --api_url http://localhost:8000/log
"""

import argparse
import time

import cv2
import numpy as np
import requests

from preprocess import bgr_frame_to_gray, preprocess_array, draw_result_overlay

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Falls back to the TF-bundled interpreter when tflite_runtime isn't
    # installed (e.g. during local dev on a laptop instead of the Jetson).
    import tensorflow as tf
    tflite = tf.lite


CLASS_NAMES = ["def_front", "ok_front"]  # index 0 / 1, alphabetical (matches training)


class TFLiteDefectDetector:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, preprocessed_img):
        batch = np.expand_dims(preprocessed_img, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], batch)
        self.interpreter.invoke()
        prob_ok = float(self.interpreter.get_tensor(self.output_details[0]["index"])[0][0])
        return prob_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera index, or -1 to use --video")
    parser.add_argument("--video", type=str, default=None, help="path to a video file instead of a live camera")
    parser.add_argument("--model", default="models/defect_cnn.tflite")
    parser.add_argument("--api_url", default=None, help="optional backend endpoint to POST results to")
    parser.add_argument("--display", action="store_true", help="show a live cv2 window (requires a GUI)")
    args = parser.parse_args()

    detector = TFLiteDefectDetector(args.model)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source")

    prev_time = time.time()
    frame_count = 0

    print("Starting real-time inspection loop. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = bgr_frame_to_gray(frame)
            processed = preprocess_array(gray)
            prob_ok = detector.predict(processed)
            is_ok = prob_ok >= 0.5
            label = "ok_front" if is_ok else "def_front"
            confidence = prob_ok if is_ok else 1 - prob_ok

            frame_count += 1
            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            if args.display:
                overlay = draw_result_overlay(frame.copy(), label, confidence, fps)
                cv2.imshow("Edge QC Inspection", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"frame={frame_count} label={label} confidence={confidence:.3f} fps={fps:.1f}")

            if args.api_url:
                try:
                    requests.post(args.api_url, json={
                        "frame": frame_count,
                        "label": label,
                        "confidence": confidence,
                        "fps": fps,
                    }, timeout=0.5)
                except requests.RequestException:
                    pass  # never let a dashboard hiccup stall the inspection line

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
