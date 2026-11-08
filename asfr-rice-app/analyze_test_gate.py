
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_PATH))

MANIFEST_PATH = (
    PROJECT_ROOT
    / "../../asfr_resume_bundle/manifests/splits/test_manifest.csv"
).resolve()

LOCK_PATH = (
    PROJECT_ROOT
    / "../../asfr_resume_bundle/reports/split_lock.json"
).resolve()

DATASET_ROOT = (
    PROJECT_ROOT / "../../Rice_Leaf_AUG"
).resolve()

MODEL_PATH = PROJECT_ROOT / "model_artifacts/best_model.keras"
CLASS_PATH = PROJECT_ROOT / "model_artifacts/class_names.json"
METADATA_PATH = PROJECT_ROOT / "model_artifacts/model_metadata.json"

OUTPUT_DIR = PROJECT_ROOT / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_IMAGE_CSV = OUTPUT_DIR / "asfr_gate_per_image.csv"
PER_CLASS_CSV = OUTPUT_DIR / "asfr_gate_by_class.csv"
SUMMARY_JSON = OUTPUT_DIR / "asfr_gate_summary.json"

EXPECTED_GATE_SHAPE = (14, 14)

# ---------------------------------------------------------
# 2. IMPORT MODEL COMPONENTS
# ---------------------------------------------------------

from app.inference import InferenceEngine
from app.xai import GradCAMExplainer


# ---------------------------------------------------------
# 3. HASH AND STATISTICS HELPERS
# ---------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def distribution_stats(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def gate_metrics(gate):
    values = np.asarray(gate, dtype=np.float64).ravel()

    return {
        "gate_min": float(np.min(values)),
        "gate_max": float(np.max(values)),
        "gate_mean": float(np.mean(values)),
        "gate_std": float(np.std(values)),
        "pct_gate_ge_095": float(np.mean(values >= 0.95) * 100),
        "pct_gate_lt_050": float(np.mean(values < 0.50) * 100),
        "pct_gate_lt_010": float(np.mean(values < 0.10) * 100),
    }


# ---------------------------------------------------------
# 4. VERIFY LOCKED MANIFEST
# ---------------------------------------------------------

print("\nASFR TEST-SET GATE ANALYSIS")
print("=" * 60)

with LOCK_PATH.open("r", encoding="utf-8") as file:
    split_lock = json.load(file)

with MANIFEST_PATH.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    manifest_rows = list(csv.DictReader(file))

manifest_hash = sha256_file(MANIFEST_PATH)
expected_manifest_hash = split_lock["manifest_sha256"]["test"]

if manifest_hash != expected_manifest_hash:
    raise RuntimeError("Test manifest hash does not match split lock.")

if split_lock.get("status") != "LOCKED":
    raise RuntimeError("Split lock does not report LOCKED status.")

if len(manifest_rows) != 1770:
    raise RuntimeError(
        f"Expected 1770 test images, found {len(manifest_rows)}."
    )

print("Manifest records:", len(manifest_rows))
print("Manifest hash verified.")
print("Split version:", split_lock.get("split_version"))

# ---------------------------------------------------------
# 5. VERIFY ALL TEST IMAGE HASHES BEFORE ANALYSIS
# ---------------------------------------------------------

print("\nVerifying image files and SHA-256 hashes...")

resolved_rows = []

for index, row in enumerate(manifest_rows, start=1):

    relative_path = row["relative_path"]

    # Map the physical folder name without editing the manifest.
    local_relative_path = relative_path.replace(
        "Neck Blast/", "Neck_Blast/", 1
    )

    image_path = DATASET_ROOT / local_relative_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Missing test image: {relative_path}"
        )

    actual_hash = sha256_file(image_path)
    expected_hash = row["sha256"].strip()

    if actual_hash.lower() != expected_hash.lower():
        raise RuntimeError(
            f"SHA-256 mismatch: {relative_path}"
        )

    resolved_rows.append((row, image_path))

    if index % 250 == 0:
        print(f"Integrity checked: {index}/{len(manifest_rows)}")

print("All 1770 image hashes verified.")

# ---------------------------------------------------------
# 6. LOAD MODEL AND ASFR GATE EXTRACTOR
# ---------------------------------------------------------

print("\nLoading inference model...")

engine = InferenceEngine(
    model_path=MODEL_PATH,
    class_names_path=CLASS_PATH,
    metadata_path=METADATA_PATH,
)

engine.load()

explainer = GradCAMExplainer(engine)

with METADATA_PATH.open("r", encoding="utf-8") as file:
    model_metadata = json.load(file)

print("Model loaded.")
print("Gate layer:", "asfr_spatial_gate")
print("Expected native gate shape:", EXPECTED_GATE_SHAPE)

# ---------------------------------------------------------
# 7. EXTRACT NATIVE GATES FOR ALL TEST IMAGES
# ---------------------------------------------------------

per_image_results = []
gates_by_class = defaultdict(list)
all_gate_values = []

print("\nExtracting native ASFR gates...")

for index, (row, image_path) in enumerate(
    resolved_rows, start=1
):

    image_bytes = image_path.read_bytes()

    # Use the same preprocessing pipeline as inference.
    _, batch = explainer._prepare(image_bytes)

    gate_batch = explainer.gate_model(
        batch,
        training=False,
    ).numpy()

    gate = np.asarray(
        gate_batch[0, ..., 0],
        dtype=np.float32,
    )

    if gate.shape != EXPECTED_GATE_SHAPE:
        raise RuntimeError(
            f"Unexpected gate shape {gate.shape}: "
            f"{row['relative_path']}"
        )

    if not np.all(np.isfinite(gate)):
        raise RuntimeError(
            f"Non-finite gate values: {row['relative_path']}"
        )

    if np.min(gate) < 0 or np.max(gate) > 1:
        raise RuntimeError(
            f"Gate values outside [0, 1]: {row['relative_path']}"
        )

    metrics = gate_metrics(gate)

    result = {
        "image_index": index,
        "relative_path": row["relative_path"],
        "class_name": row["class_name"],
        **metrics,
    }

    per_image_results.append(result)

    flattened_gate = gate.ravel().astype(np.float64)

    gates_by_class[row["class_name"]].append(
        flattened_gate
    )

    all_gate_values.append(flattened_gate)

    if index % 50 == 0 or index == len(resolved_rows):
        print(
            f"Processed {index}/{len(resolved_rows)} "
            f"test images"
        )

# ---------------------------------------------------------
# 8. SAVE PER-IMAGE RESULTS
# ---------------------------------------------------------

print("\nSaving per-image results...")

with PER_IMAGE_CSV.open(
    "w", encoding="utf-8", newline=""
) as file:

    fieldnames = list(per_image_results[0].keys())

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(per_image_results)

# ---------------------------------------------------------
# 9. CALCULATE CLASS-WISE DISTRIBUTIONS
# ---------------------------------------------------------

class_results = []

for class_name in sorted(gates_by_class):

    class_values = np.concatenate(
        gates_by_class[class_name]
    )

    class_image_rows = [
        result
        for result in per_image_results
        if result["class_name"] == class_name
    ]

    class_result = {
        "class_name": class_name,
        "image_count": len(class_image_rows),
        **distribution_stats(class_values),
        "mean_image_gate_mean": float(
            np.mean([
                item["gate_mean"]
                for item in class_image_rows
            ])
        ),
        "mean_image_gate_std": float(
            np.mean([
                item["gate_std"]
                for item in class_image_rows
            ])
        ),
        "mean_pct_gate_ge_095": float(
            np.mean([
                item["pct_gate_ge_095"]
                for item in class_image_rows
            ])
        ),
        "mean_pct_gate_lt_050": float(
            np.mean([
                item["pct_gate_lt_050"]
                for item in class_image_rows
            ])
        ),
    }

    class_results.append(class_result)

with PER_CLASS_CSV.open(
    "w", encoding="utf-8", newline=""
) as file:

    fieldnames = list(class_results[0].keys())

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(class_results)

# ---------------------------------------------------------
# 10. OVERALL GATE DISTRIBUTION
# ---------------------------------------------------------

all_values = np.concatenate(all_gate_values)

overall_distribution = distribution_stats(all_values)

overall_image_means = np.array([
    item["gate_mean"]
    for item in per_image_results
])

overall_image_stds = np.array([
    item["gate_std"]
    for item in per_image_results
])

overall_saturation = np.array([
    item["pct_gate_ge_095"]
    for item in per_image_results
])

overall_low_gate = np.array([
    item["pct_gate_lt_050"]
    for item in per_image_results
])

summary = {
    "analysis": "Full locked-test-set ASFR spatial gate analysis",
    "test_images": len(per_image_results),
    "native_gate_shape": list(EXPECTED_GATE_SHAPE),
    "gate_layer": "asfr_spatial_gate",
    "manifest_sha256": manifest_hash,
    "split_version": split_lock.get("split_version"),
    "model_metadata": model_metadata,
    "overall_gate_value_distribution": overall_distribution,
    "per_image_statistics": {
        "mean_gate_mean": float(np.mean(overall_image_means)),
        "std_gate_mean": float(np.std(overall_image_means)),
        "mean_within_image_gate_std": float(
            np.mean(overall_image_stds)
        ),
        "mean_percentage_gate_ge_095": float(
            np.mean(overall_saturation)
        ),
        "median_percentage_gate_ge_095": float(
            np.median(overall_saturation)
        ),
        "mean_percentage_gate_lt_050": float(
            np.mean(overall_low_gate)
        ),
    },
    "class_count": len(class_results),
    "class_names": [
        item["class_name"]
        for item in class_results
    ],
    "interpretation_note": (
        "Gate values represent learned spatial gating values. "
        "They are not class-specific Grad-CAM scores, lesion "
        "masks, or direct evidence of causal feature importance. "
        "This is descriptive analysis, not model tuning."
    ),
}

with SUMMARY_JSON.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

# ---------------------------------------------------------
# 11. PRINT FINAL REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ASFR SPATIAL-GATE ANALYSIS COMPLETE")
print("=" * 60)

print("Images analyzed:", len(per_image_results))
print("Native gate shape:", EXPECTED_GATE_SHAPE)

print("\nOVERALL GATE DISTRIBUTION")
for key, value in overall_distribution.items():
    print(f"{key}: {value:.6f}" if isinstance(value, float)
          else f"{key}: {value}")

print("\nPER-IMAGE SUMMARY")
for key, value in summary["per_image_statistics"].items():
    print(f"{key}: {value:.6f}")

print("\nCLASS-WISE SUMMARY")

for item in class_results:
    print(
        f"{item['class_name']}: "
        f"images={item['image_count']}, "
        f"mean={item['mean']:.4f}, "
        f"std={item['std']:.4f}, "
        f"gate>=0.95={item['mean_pct_gate_ge_095']:.2f}%, "
        f"gate<0.50={item['mean_pct_gate_lt_050']:.2f}%"
    )

print("\nOUTPUT FILES")
print(PER_IMAGE_CSV)
print(PER_CLASS_CSV)
print(SUMMARY_JSON)

print("\nAll analysis outputs saved successfully.")
