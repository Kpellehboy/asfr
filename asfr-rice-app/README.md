# ASFR Rice Disease Classification — Research Demo

A research-prototype web application for **image-level, nine-class rice disease classification** using a ResNet50V2 + ASFR model. It presents a class-probability distribution, prediction-conditioned Grad-CAM heatmap/overlay, and the ASFR spatial-gate visualization.

> **Research limitation:** This is not a field-validated agricultural diagnostic service. Grad-CAM and the ASFR gate are model visualizations, not lesion segmentation masks or proof of causal reasoning. Benchmark metrics do not establish field robustness. External/OOD validation remains pending.

## Architecture

- `backend/`: FastAPI inference and explanation API.
- `frontend/`: Next.js 15 / React 19 interface with a dark, dimensional, emerald 3D-inspired visual system.
- `model_artifacts/`: class mapping and model metadata contract. The trained `.keras` checkpoint is deliberately not included in this source archive.
- `backend/app/xai.py`: Grad-CAM target `asfr_fusion_relu`; ASFR gate target `asfr_spatial_gate`.

## Required model artifact

Place the final notebook-selected checkpoint here:

`model_artifacts/best_model.keras`

Then update `model_artifacts/model_metadata.json` from the **same final notebook export**, including:
- exact checkpoint SHA-256 in `artifact_sha256`;
- exact model version;
- exact class names in label-index order;
- input shape `[null, 224, 224, 3]`;
- output shape `[null, 9]`;
- verified parameter count and framework versions.

The backend checks the checkpoint checksum when `artifact_sha256` is present and rejects a mismatch. It also checks that the class order in metadata matches `class_names.json`. Never guess or manually reorder the classes: export the mapping from the notebook that created the selected checkpoint.

## Local development

### 1. Backend

Use Python 3.11 in a clean virtual environment (the Dockerfile uses Python 3.11).

Windows PowerShell:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..
$env:MODEL_ARTIFACT_PATH = (Resolve-Path "model_artifacts/best_model.keras").Path
$env:CLASS_NAMES_PATH = (Resolve-Path "model_artifacts/class_names.json").Path
$env:MODEL_METADATA_PATH = (Resolve-Path "model_artifacts/model_metadata.json").Path
$env:ALLOWED_ORIGINS = "http://localhost:3000"
uvicorn app.main:app --reload --app-dir backend --port 8000
```

If you run the command from inside `backend`, use `uvicorn app.main:app --reload --port 8000` instead.

### 2. Frontend

Open a second terminal:

```powershell
cd frontend
npm ci
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:3000`.

### 3. Docker Compose

After adding the final model checkpoint and updating metadata:

```bash
docker compose up --build
```

Frontend: `http://localhost:3001`  
Backend API: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

## API endpoints

- `GET /health` — process health.
- `GET /ready` — model/XAI readiness.
- `GET /model-info` — model version, input/output shape and class mapping.
- `POST /predict` — classification only.
- `POST /explain` — Grad-CAM explanation.
- `POST /analyze` — combined prediction, Grad-CAM, and ASFR gate output.

Upload form field: `file`.

## Verification before release

Run from the project root:

```powershell
python -m pytest backend/tests -q
```

Then run the real-checkpoint smoke test:

```powershell
python backend/smoke_test_real_model.py
```

Run frontend checks:

```powershell
cd frontend
npm ci
npm run typecheck
npm run build
```

The source archive does **not** include the final `.keras` checkpoint. Therefore real model loading, checkpoint checksum validation, prediction parity, XAI parity, and end-to-end integration tests remain unverified until the exact checkpoint from the final notebook is installed. Do not describe this archive as fully production-tested.

## Research/model synchronization checklist

1. Copy the selected ASFR checkpoint from the final notebook to `model_artifacts/best_model.keras`.
2. Export the exact class order and metadata from that same notebook run.
3. Confirm `asfr_fusion_relu` and `asfr_spatial_gate` exist in the saved model.
4. Run the real-checkpoint smoke test.
5. Run API integration tests, especially `/analyze`, explicit Grad-CAM target class, malformed uploads, upload-size enforcement, and concurrent requests.
6. Confirm the frontend model version and returned nine-class order match the notebook.
7. Keep OOD status marked pending/experimental until an independent OOD evaluation has actually been completed.
