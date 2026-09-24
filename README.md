# IRONSIGHT — Automated Vision QC for Manufacturing

An end-to-end automated quality-control system: a CNN trained on real
casting-product images classifies parts as **PASS** or **DEFECT**, exported
to TensorFlow Lite for real-time inference on an **NVIDIA Jetson** edge
device, with a FastAPI backend and an animated web dashboard + chat
assistant for line operators. Also deployable as a hosted web dashboard
(e.g. on Render), with sample images served from Google Cloud Storage.

Built on the [casting product image dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(submersible pump impellers, defective vs. ok, ~7,300 images).

```
ironsight-vision-qc/
├── data/                        # extracted dataset (train/ + test/ splits)
│   └── casting_data/
│       └── casting_data/
│           ├── train/
│           │   ├── def_front/
│           │   └── ok_front/
│           └── test/
│               ├── def_front/
│               └── ok_front/
├── src/
│   ├── model.py                # CNN architecture
│   ├── train.py                # training + evaluation + export pipeline
│   ├── preprocess.py            # shared OpenCV preprocessing (train == inference)
│   ├── inference.py             # single-image CLI inference
│   └── run_camera.py            # real-time camera loop for the Jetson
├── app/
│   ├── backend/app.py            # FastAPI: /api/predict /api/stats /api/chat /api/sample_image
│   └── frontend/                 # index.html / styles.css / app.js dashboard
├── docker/
│   ├── Dockerfile                # x86 dashboard/server image
│   ├── Dockerfile.jetson         # arm64 / L4T image for the edge unit
│   └── docker-compose.yml
├── models/                     # defect_cnn.keras + defect_cnn.tflite (generated)
├── reports/                    # training curves, confusion matrix, metrics.json
├── runtime.txt / .python-version  # pins the Python version for hosting platforms
└── requirements.txt
```

## 1. Train the model

```bash
pip install -r requirements.txt
python src/train.py --data_dir data/casting_data/casting_data --epochs 12
```

This trains a ~260K-parameter CNN (4 conv blocks + global average pooling),
evaluates it on the held-out test split, and writes:

- `models/defect_cnn.keras` — full model
- `models/defect_cnn.tflite` — INT8-quantized model for edge inference
- `reports/training_curves.png`, `reports/confusion_matrix.png`, `reports/metrics.json`

## 2. Run the dashboard locally

```bash
cd app/backend
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` — drag a part image onto the inspection panel
to get a live PASS/DEFECT verdict, watch the session stats animate, and
chat with the line assistant about the model or the current part.

By default, the "Try a sample defect" button pulls a random real image from
the local `data/casting_data/casting_data/test/` folder. To test it against
a specific class, drag images directly from `test/def_front/` or
`test/ok_front/` onto the inspection panel.

## 3. Run it in Docker (server / demo box)

```bash
docker compose -f docker/docker-compose.yml up --build
```

## 4. Deploy to an NVIDIA Jetson (edge, real-time camera)

```bash
# on the Jetson, after flashing JetPack and installing Docker + nvidia-docker:
docker build -f docker/Dockerfile.jetson -t ironsight-edge .
docker run --runtime nvidia --device /dev/video0 ironsight-edge --camera 0
```

`run_camera.py` reads frames straight from the inspection-line camera,
classifies each one with the quantized TFLite model, and overlays a
PASS/DEFECT HUD — the same script also accepts `--api_url` to stream verdicts
back to the dashboard for logging.

## 5. Deploy the dashboard to Render (hosted web demo)

The dashboard can also run as a standard hosted web service (e.g. for demos,
stakeholder reviews, or anywhere Jetson hardware isn't available), using
[Render](https://render.com) as the host and **Google Cloud Storage** to
serve sample images (since the full dataset isn't committed to the repo).

**One-time setup:**

1. Push this repo to GitHub. Note that `data/`, `casting_512x512/`,
   `venv/`, and other large/local-only folders are excluded via
   `.gitignore` — only code, configs, and the small `models/` files are
   committed.
2. Create a GCS bucket (e.g. `your-project-data`) and upload the
   `test/def_front` and `test/ok_front` folders into it, preserving them
   as top-level folders in the bucket (`def_front/`, `ok_front/`).
3. Create a GCS service account with **Storage Object Viewer** access
   scoped to just that bucket, and download its JSON key.
4. On Render, create a new Web Service pointing at this GitHub repo:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.backend.app:app --host 0.0.0.0 --port $PORT`
5. In Render's Environment tab, set:
   - `PYTHON_VERSION` — e.g. `3.12.10`, matching your local dev version
     (needed because `tensorflow-cpu` doesn't yet support the very latest
     Python releases)
   - `SAMPLE_IMAGES_BUCKET` — your GCS bucket name
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON` — paste the entire contents of
     the service account's JSON key as the value

Once deployed, `/api/sample_image` automatically serves from the GCS
bucket instead of the local `data/` folder — no code changes needed, this
is handled by `app.py` checking for `SAMPLE_IMAGES_BUCKET` at startup.

**Note:** whichever folder you upload to GCS is what "Try a sample defect"
will draw from — make sure it's the same `test/def_front` and
`test/ok_front` folders the model was actually trained and evaluated on,
not a differently-sized or differently-sourced copy of similar images, or
prediction quality will look artificially poor even though the model
itself is fine.

## Why these design choices

- **Grayscale, 128×128 input** — casting surface defects (blow holes, burrs,
  shrinkage) show up as texture/edge irregularities that don't need color;
  a smaller grayscale input keeps the model fast enough for Jetson Nano-class
  hardware.
- **Same `preprocess.py` for training and inference** — a very common source
  of "it worked in the notebook but not on the device" bugs is a mismatch
  between train-time and inference-time preprocessing (resize method,
  normalization, color order). This project shares one module for both.
- **TFLite + quantization for the edge** — cuts model size and latency
  substantially with a small accuracy trade-off, verified against the same
  test set as the full Keras model in `reports/metrics.json`.
- **Rule-based chat assistant by default** — keeps the whole stack usable on
  an air-gapped factory network with no external API dependency. Swap
  `answer_chat()` in `app/backend/app.py` for a call to the Anthropic API if
  you want open-ended Q&A and have outbound internet access from the line.

## Retraining on your own defect data

Point `--data_dir` at any folder shaped like:

```
your_data/
├── train/
│   ├── def_front/   (or any defect-class folder name)
│   └── ok_front/
└── test/
    ├── def_front/
    └── ok_front/
```

and rerun `train.py`. No code changes needed for a new binary defect-detection
task; for multi-class defects (e.g. distinguishing burr vs. blow-hole vs. ok),
change the final Dense layer in `model.py` from a single sigmoid output to a
softmax output with one unit per class, switch the loss in `train.py` from
`binary_crossentropy` to `categorical_crossentropy` (or
`sparse_categorical_crossentropy` if labels are integer-encoded), and update
`CLASS_NAMES` and the label-decoding logic in `app.py` / `inference.py` /
`run_camera.py` accordingly.
