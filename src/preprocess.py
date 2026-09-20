"""
preprocess.py
-------------
OpenCV-based preprocessing pipeline shared by training, inference, and
the live-camera edge pipeline. Keeping this logic in one place ensures
the exact same transform is used at train time and at inference time
on the Jetson (a very common source of silent accuracy loss when
train/inference preprocessing drifts apart).
"""

import cv2
import numpy as np

IMG_SIZE = (128, 128)


def load_and_preprocess(image_path, img_size=IMG_SIZE):
    """Load an image from disk with OpenCV and prepare it for the model.

    IMPORTANT: this must produce numerically the same tensor the model saw
    during training. `train.py` builds its datasets with
    `tf.keras.utils.image_dataset_from_directory(..., color_mode="grayscale",
    image_size=img_size)`, which just decodes to grayscale and resizes with
    bilinear interpolation -- no denoising/contrast steps, because the
    model's own `Rescaling(1/255)` layer handles normalization internally.
    This function mirrors exactly that: any extra enhancement (CLAHE, blur,
    etc.) shifts the input distribution away from what the network was
    trained on and silently destroys accuracy, especially on the OK class.
    If you add preprocessing here, add the *same* step to the tf.data
    pipeline in train.py's `make_datasets()` and retrain -- never change
    one without the other.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return preprocess_array(img, img_size)


def preprocess_array(img_gray, img_size=IMG_SIZE):
    """Same pipeline as load_and_preprocess but for an in-memory frame
    (used by the real-time camera loop, where frames come from cv2.VideoCapture
    rather than disk). Bilinear resize only -- see the docstring above for why."""
    img_gray = cv2.resize(img_gray, img_size, interpolation=cv2.INTER_LINEAR)
    img_gray = img_gray.astype(np.float32)
    img_gray = np.expand_dims(img_gray, axis=-1)  # (H, W, 1)
    return img_gray


def bgr_frame_to_gray(frame):
    """Convert a BGR camera frame (as returned by cv2.VideoCapture) to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def draw_result_overlay(frame_bgr, label, confidence, fps=None):
    """Draws a QC-style HUD overlay on a frame: PASS/FAIL banner + confidence + fps.
    Used by the live edge-deployment demo (run_camera.py) so operators on the
    factory floor get an immediate visual verdict, matching how the deployed
    Jetson unit would present results on a line-side monitor.
    """
    h, w = frame_bgr.shape[:2]
    is_defect = label.lower().startswith("def")
    color = (0, 0, 255) if is_defect else (0, 200, 0)  # BGR: red for defect, green for ok
    banner_text = "DEFECT DETECTED" if is_defect else "PASS"

    cv2.rectangle(frame_bgr, (0, 0), (w, 40), color, thickness=-1)
    cv2.putText(frame_bgr, f"{banner_text}  ({confidence*100:.1f}%)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(frame_bgr, f"{fps:.1f} FPS", (w - 120, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr
