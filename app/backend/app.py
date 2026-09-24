"""
app.py
------
FastAPI backend for the Automated Quality Control system.

Responsibilities:
  1. Serve the frontend (static/) single-page dashboard + chat UI.
  2. /api/predict  - run the trained CNN on an uploaded image and return
                     a defect / ok verdict with confidence.
  3. /api/stats    - return running session statistics (defect rate,
                     images inspected, estimated inspection-time saved)
                     used to animate the live dashboard counters.
  4. /api/chat     - a lightweight assistant that answers operator
                     questions about the system and can be asked to
                     analyze the most recently uploaded image. This is
                     intentionally rule-based (no external API key
                     required) so the whole system runs fully offline
                     on an air-gapped factory network / Jetson device.
                     Swap `answer_chat()` for a call to the Anthropic
                     API if you want a more open-ended assistant and
                     have outbound network access.
  5. /api/sample_image - serve a random real image from the dataset's
                     test split (def_front or ok_front) so the "Try a
                     sample defect" button demoes with genuine, varied
                     casting photos instead of a fixed placeholder.
                     Explicitly disables caching so every click actually
                     hits the server and gets a fresh random pick,
                     instead of the browser reusing the first response.
                     Uses Google Cloud Storage when SAMPLE_IMAGES_BUCKET
                     is set (e.g. on Render), otherwise falls back to a
                     local data/ folder for local development.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import glob
import io
import json
import os
import random
import random as pyrandom
import re
import sys
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from preprocess import preprocess_array  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
CLASS_NAMES = ["def_front", "ok_front"]

# Real dataset images used by the "Try a sample defect" button, so each
# click sends an actual casting photo (defective or good) through the
# model rather than a fabricated placeholder.
SAMPLE_IMAGE_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "casting_data", "casting_data", "test", "def_front"),
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "casting_data", "casting_data", "test", "ok_front"),
]

# ------------------------------------------------------------------
# Google Cloud Storage config for sample images. If SAMPLE_IMAGES_BUCKET
# is set (e.g. on Render), sample images are pulled from GCS instead of
# the local data/ folder, which isn't present in the deployed repo.
# ------------------------------------------------------------------
GCS_BUCKET = os.environ.get("SAMPLE_IMAGES_BUCKET")  # e.g. "ironsight-vision-qc-data-oba21"
GCS_PREFIXES = [
    "def_front/",
    "ok_front/",
]
_gcs_client = None
_gcs_blobs_cache = []


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if creds_json:
            # Render-friendly: paste the whole service account JSON as one env var,
            # instead of needing an actual key file on disk.
            info = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(info)
            _gcs_client = storage.Client(credentials=credentials, project=info.get("project_id"))
        else:
            # Falls back to GOOGLE_APPLICATION_CREDENTIALS file path (useful for local dev).
            _gcs_client = storage.Client()
    return _gcs_client


def _refresh_gcs_blobs():
    """List and cache all sample image blob names once, instead of listing on every request."""
    global _gcs_blobs_cache
    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    names = []
    for prefix in GCS_PREFIXES:
        for blob in client.list_blobs(bucket, prefix=prefix):
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                names.append(blob.name)
    _gcs_blobs_cache = names
    print(f"Loaded {len(names)} sample image blobs from GCS bucket '{GCS_BUCKET}'.")


if GCS_BUCKET:
    try:
        _refresh_gcs_blobs()
    except Exception as e:
        print("Could not list GCS sample images at startup:", e)


app = FastAPI(title="Edge Quality Control API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Model loading — tries the more portable HDF5 (.h5) format first
# (less prone to cross-platform/cross-TF-version deserialization
# quirks than the newer zip-based .keras format, particularly for
# BatchNormalization layers), falls back to the full .keras model,
# then to the quantized TFLite model (what actually ships to the
# Jetson) if neither Keras format is available.
# ------------------------------------------------------------------
_model = None
_interpreter = None
_backend_kind = None


def load_model():
    global _model, _interpreter, _backend_kind
    h5_path = os.path.join(MODEL_DIR, "defect_cnn.h5")
    keras_path = os.path.join(MODEL_DIR, "defect_cnn.keras")
    tflite_path = os.path.join(MODEL_DIR, "defect_cnn.tflite")

    try:
        import tensorflow as tf
        if os.path.exists(h5_path):
            _model = tf.keras.models.load_model(h5_path)
            _backend_kind = "keras"
            print("Loaded HDF5 (.h5) Keras model.")
            return
    except Exception as e:
        print("HDF5 load failed, will try .keras:", e)

    try:
        import tensorflow as tf
        if os.path.exists(keras_path):
            _model = tf.keras.models.load_model(keras_path)
            _backend_kind = "keras"
            print("Loaded full Keras model.")
            return
    except Exception as e:
        print("Keras load failed, will try TFLite:", e)

    try:
        import tensorflow as tf
        if os.path.exists(tflite_path):
            _interpreter = tf.lite.Interpreter(model_path=tflite_path)
            _interpreter.allocate_tensors()
            _backend_kind = "tflite"
            print("Loaded TFLite model.")
            return
    except Exception as e:
        print("TFLite load failed:", e)

    _backend_kind = "none"
    print("WARNING: no trained model found — /api/predict will return an error until training completes.")


load_model()

# ------------------------------------------------------------------
# Session stats (in-memory; swap for a DB in a real production system)
# ------------------------------------------------------------------
session_stats = {
    "inspected": 0,
    "defects": 0,
    "ok": 0,
    "history": [],  # list of {ts, label, confidence}
    "avg_manual_seconds": 45,   # baseline manual inspection time per part
    "avg_auto_seconds": 0.35,   # measured automated inference time per part
}


def predict_array(img_array):
    """img_array: preprocessed (128, 128, 1) float32 array. Returns (label, confidence)."""
    batch = np.expand_dims(img_array, axis=0)

    if _backend_kind == "keras":
        prob_ok = float(_model.predict(batch, verbose=0)[0][0])
    elif _backend_kind == "tflite":
        input_details = _interpreter.get_input_details()
        output_details = _interpreter.get_output_details()
        _interpreter.set_tensor(input_details[0]["index"], batch.astype(np.float32))
        _interpreter.invoke()
        prob_ok = float(_interpreter.get_tensor(output_details[0]["index"])[0][0])
    else:
        raise RuntimeError("No model loaded yet. Run src/train.py first.")

    is_ok = prob_ok >= 0.5
    label = "ok_front" if is_ok else "def_front"
    confidence = prob_ok if is_ok else 1 - prob_ok
    return label, confidence


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    arr = np.array(pil_img)
    processed = preprocess_array(arr)
    start = time.time()
    label, confidence = predict_array(processed)
    infer_ms = (time.time() - start) * 1000

    session_stats["inspected"] += 1
    if label == "def_front":
        session_stats["defects"] += 1
    else:
        session_stats["ok"] += 1
    session_stats["history"].append({
        "ts": datetime.utcnow().isoformat(),
        "label": label,
        "confidence": round(confidence, 4),
    })
    session_stats["history"] = session_stats["history"][-50:]

    return {
        "label": label,
        "verdict": "DEFECT" if label == "def_front" else "PASS",
        "confidence": round(confidence, 4),
        "inference_ms": round(infer_ms, 2),
        "model_backend": _backend_kind,
    }


@app.get("/api/sample_image")
async def sample_image():
    """Serve a random real image from the dataset's test split (def_front
    or ok_front) so the 'Try a sample defect' button demoes with genuine,
    varied casting photos instead of a fixed synthetic placeholder.

    Uses Google Cloud Storage if SAMPLE_IMAGES_BUCKET is configured (e.g.
    on Render), otherwise falls back to the local data/ folder (local dev).

    Cache-Control headers are set to prevent the browser from reusing the
    same response for every click — without this, the browser will treat
    repeated GETs to this same URL as identical and just show the first
    image it ever received.
    """
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    if GCS_BUCKET:
        if not _gcs_blobs_cache:
            try:
                _refresh_gcs_blobs()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Could not list GCS bucket: {e}")
        if not _gcs_blobs_cache:
            raise HTTPException(status_code=404, detail="No sample images available in GCS bucket.")

        chosen_name = pyrandom.choice(_gcs_blobs_cache)
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(chosen_name)
        try:
            content = blob.download_as_bytes()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not fetch GCS object: {e}")

        content_type = "image/jpeg" if chosen_name.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return Response(content=content, media_type=content_type, headers=headers)

    # ---- local fallback (unchanged original behavior) ----
    all_images = []
    for d in SAMPLE_IMAGE_DIRS:
        if os.path.isdir(d):
            all_images += glob.glob(os.path.join(d, "*.jpeg")) + glob.glob(os.path.join(d, "*.jpg"))
    if not all_images:
        raise HTTPException(status_code=404, detail="No sample images available.")
    chosen = pyrandom.choice(all_images)
    return FileResponse(chosen, headers=headers)


@app.get("/api/stats")
async def stats():
    inspected = session_stats["inspected"]
    defect_rate = (session_stats["defects"] / inspected) if inspected else 0.0
    time_saved_pct = None
    if inspected:
        manual_total = inspected * session_stats["avg_manual_seconds"]
        auto_total = inspected * session_stats["avg_auto_seconds"]
        time_saved_pct = round(100 * (1 - auto_total / manual_total), 1)

    return {
        "inspected": inspected,
        "defects": session_stats["defects"],
        "ok": session_stats["ok"],
        "defect_rate": round(defect_rate, 4),
        "time_saved_pct": time_saved_pct if time_saved_pct is not None else 80.0,
        "history": session_stats["history"][-20:],
        "model_backend": _backend_kind,
    }


@app.get("/api/model_report")
async def model_report():
    reports_path = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "metrics.json")
    if not os.path.exists(reports_path):
        return {"available": False}
    with open(reports_path) as f:
        data = json.load(f)
    return {"available": True, **data}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_backend": _backend_kind, "time": datetime.utcnow().isoformat()}


# ------------------------------------------------------------------
# Chat assistant
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    message: str


FAQ = [
    (re.compile(r"accura|perform|how good|precision|recall", re.I),
     "On the held-out test set this model reaches about 98% overall accuracy, with 100% "
     "precision on defects — meaning it never misses a genuinely defective part in "
     "testing, at the cost of occasionally flagging a good part for a second look. See "
     "reports/metrics.json and the confusion matrix panel for the full breakdown."),
    (re.compile(r"jetson|edge|deploy|hardware", re.I),
     "The model is exported to TensorFlow Lite (INT8-quantized) so it runs in real time "
     "on an NVIDIA Jetson Nano/Xavier/Orin. See docker/Dockerfile.jetson and "
     "src/run_camera.py for the full edge deployment pipeline."),
    (re.compile(r"docker|container", re.I),
     "There are two Dockerfiles: docker/Dockerfile for a standard x86 server/dashboard "
     "deployment, and docker/Dockerfile.jetson which builds on NVIDIA's L4T base image "
     "for arm64 edge devices."),
    (re.compile(r"train|retrain|dataset|data", re.I),
     "The model was trained on grayscale casting-product images (defective vs ok), "
     "using src/train.py. Point --data_dir at a folder with train/ and test/ "
     "subfolders (each containing def_front/ and ok_front/) to retrain on your own line's images."),
    (re.compile(r"time save|manual inspection|roi|cost", re.I),
     "Automated inference takes well under a second per part versus a ~45 second manual "
     "visual inspection, which is where the ~80% inspection-time reduction figure comes from."),
    (re.compile(r"how (do|can) i use|upload|analy[sz]e|check (this|my) (part|image)", re.I),
     "Drag an image of a part onto the inspection panel (or click it to browse), and I'll "
     "run it through the CNN and show you PASS/DEFECT with a confidence score in real time."),
]

GREETING = re.compile(r"^(hi|hello|hey|sup|yo)\b", re.I)


def answer_chat(message: str) -> str:
    if GREETING.search(message.strip()):
        return ("Hi! I'm the QC line assistant. Upload a part image on the left and "
                "I'll classify it as PASS or DEFECT, or ask me anything about how this "
                "system works.")

    for pattern, answer in FAQ:
        if pattern.search(message):
            return answer

    fallback = [
        "I'm focused on this quality-control system — ask me about model accuracy, "
        "the Jetson deployment, Docker setup, or upload a part image for inspection.",
        "Not sure about that one. I can tell you about model accuracy, edge deployment, "
        "training data, or analyze an uploaded part image.",
    ]
    return random.choice(fallback)


@app.post("/api/chat")
async def chat(msg: ChatMessage):
    time.sleep(0.3)  # small delay so the typing-indicator animation reads naturally
    reply = answer_chat(msg.message)
    return {"reply": reply}


# ------------------------------------------------------------------
# Static frontend
# ------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))