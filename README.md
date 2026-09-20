# IRONSIGHT — Automated Vision QC for Manufacturing

An end-to-end automated quality-control system: a CNN trained on real
casting-product images classifies parts as **PASS** or **DEFECT**, exported
to TensorFlow Lite for real-time inference on an **NVIDIA Jetson** edge
device, with a FastAPI backend and an animated web dashboard + chat
assistant for line operators.

Built on the [casting product image dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(submersible pump impellers, defective vs. ok, ~7,300 images).

```
qc_project/
├── data_raw/                  # extracted dataset (train/ + test/ splits)
├── src/
│   ├── model.py                # CNN architecture
│   ├── train.py                # training + evaluation + export pipeline
│   ├── preprocess.py            # shared OpenCV preprocessing (train == inference)
│   ├── inference.py             # single-image CLI inference
│   └── run_camera.py            # real-time camera loop for the Jetson
├── app/
│   ├── backend/app.py            # FastAPI: /api/predict /api/stats /api/chat
│   └── frontend/                 # index.html / styles.css / app.js dashboard
├── docker/
│   ├── Dockerfile                # x86 dashboard/server image
│   ├── Dockerfile.jetson         # arm64 / L4T image for the edge unit
│   └── docker-compose.yml
├── models/                     # defect_cnn.keras + defect_cnn.tflite (generated)
├── reports/                    # training curves, confusion matrix, metrics.json
└── requirements.txt
```

## 1. Train the model

```bash
pip install -r requirements.txt
python src/train.py --data_dir data_raw/casting_data/casting_data --epochs 12
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
change the final Dense layer to `softmax` with `n_classes` units and the loss
to `categorical_crossentropy` in `src/model.py`.
