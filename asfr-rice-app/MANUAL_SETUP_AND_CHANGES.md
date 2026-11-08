# Manual Setup After Downloading the Refined ASFR App

This project archive contains the refined source code. It does **not** contain the trained TensorFlow `.keras` checkpoint because the uploaded GitHub ZIP did not include that large model file.

## 1. Add the final notebook checkpoint — required

From your final Kaggle notebook output, download the selected ASFR model checkpoint and place it at:

`model_artifacts/best_model.keras`

Use the selected model from the final notebook, not an older checkpoint or a different seed. The app's Grad-CAM and gate visualizations depend on the exact architecture and layer names in that saved model.

## 2. Update model metadata — required

Open:

`model_artifacts/model_metadata.json`

Replace the explicit `REPLACE_...` / `PENDING` values using the final notebook's exported metadata. At minimum verify:

- `model_name`: the selected ASFR model identity.
- `model_version`: a version you choose, e.g. `asfr-rice-final-seed-<seed>-v1`.
- `artifact`: `best_model.keras`.
- `artifact_sha256`: SHA-256 of the exact `.keras` file you copied.
- `class_names_in_label_order`: the exact output-index order from the final notebook.
- `parameter_count`: exact count printed by the notebook.
- framework/Keras versions used to save the checkpoint.

Do not manually guess the class order. `class_names_in_label_order` must exactly equal `class_names` in `model_artifacts/class_names.json`. The backend now checks this and checks the checkpoint SHA-256.

To calculate the checkpoint SHA-256 on Windows PowerShell:

```powershell
Get-FileHash .\model_artifacts\best_model.keras -Algorithm SHA256
```

Copy the `Hash` value into `artifact_sha256` in `model_metadata.json`.

## 3. Confirm class labels and order — required

Open both:
- `model_artifacts/class_names.json`
- `model_artifacts/model_metadata.json`

Confirm the nine names and their order match the final notebook's `class_names` / label encoder. The current display labels are:

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf Scald
6. Narrow Brown Leaf Spot
7. Neck Blast
8. Rice Hispa
9. Sheath Blight

If the final notebook uses a different spelling or order, update both files consistently. **Do not change order only to improve presentation**: output index is tied to the model's training labels.

## 4. Verify architecture-specific XAI layers — required

The backend currently expects these exact layer names:

- Grad-CAM target: `asfr_fusion_relu`
- ASFR gate target: `asfr_spatial_gate`

Check these names exist in `best_model.keras`. If the final checkpoint uses different names, update the constants at the top of `backend/app/xai.py` to the exact saved-model layer names, then rerun XAI tests.

The ASFR gate is a spatial-gating visualization. It is not a lesion mask. Grad-CAM is a prediction-conditioned feature attribution visualization, not proof of causal reasoning.

## 5. Run the backend

Recommended: Python 3.11, matching the Docker base image.

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --app-dir backend --port 8000
```

If the model files are in the expected paths, the API loads the model at startup. Open:
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Readiness: `http://127.0.0.1:8000/ready`
- Model info: `http://127.0.0.1:8000/model-info`

If the backend reports that the model cannot be initialized, check the checkpoint path, checksum, class mapping, and TensorFlow/Keras versions.

## 6. Run the frontend

Open a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. The default frontend API URL is `http://127.0.0.1:8000`.

If you deploy the backend elsewhere, create `frontend/.env.local` with:

```env
NEXT_PUBLIC_API_BASE_URL=https://YOUR-BACKEND-HOST
```

Replace the placeholder with your actual backend URL. Do not add secrets to `NEXT_PUBLIC_*` variables.

## 7. Run verification before pushing to GitHub

From the project root:

```powershell
python -m pytest backend/tests -q
python backend/smoke_test_real_model.py
```

Frontend:

```powershell
cd frontend
npm ci
npm run typecheck
npm run build
```

Prioritize these real-checkpoint tests:
- model loads and matches the checkpoint SHA-256;
- `/model-info` returns the correct class order and model version;
- `/predict` returns nine probabilities summing to approximately 1;
- `/analyze` returns prediction, Grad-CAM overlay/heatmap, and ASFR gate;
- Grad-CAM and gate layers are present in the final checkpoint;
- invalid/corrupt uploads and oversized files are rejected;
- frontend displays the actual API response.

## 8. Before publishing the refined repository

- Keep the final model checkpoint out of Git if it is too large; use Git LFS or a versioned model-artifact release if appropriate.
- Never commit `.env`, API secrets, `.venv`, `node_modules`, `.next`, or cache files.
- Update README model version and dataset/benchmark claims only from verified notebook outputs.
- Keep OOD status as pending until you have completed and documented the independent OOD experiment.
- Describe the app as a research prototype, not a field-validated diagnostic tool.

## What I changed in this archive

- Reworked the frontend visual system into a dark emerald, glass-like, 3D-inspired research dashboard while preserving the existing upload, analysis, prediction, probability, Grad-CAM, and gate interactions.
- Made backend model paths configurable with environment variables.
- Fixed checkpoint hash lookup to recognize the metadata field `artifact_sha256`.
- Added class-order consistency validation between model metadata and `class_names.json`.
- Made CORS origins configurable through `ALLOWED_ORIGINS`.
- Aligned the Docker Compose backend port with the frontend's default API URL.
- Rewrote README setup, architecture, endpoint, validation, and research-limitation guidance.
- Excluded local build/dependency directories and Git history from the handoff package.

## Verification status / known limitation

Python source syntax compilation passed. The frontend dependency installation could not complete in the packaging environment, so TypeScript/build verification was not completed. The `.keras` checkpoint is absent from the uploaded repository archive; therefore model loading, inference parity, Grad-CAM, gate extraction, and full end-to-end tests still require your final notebook checkpoint.
