
import os
import json
import hashlib
import tensorflow as tf
import keras

MODEL_PATH = "model_artifacts/best_model.keras"
METADATA_PATH = "model_artifacts/model_metadata.json"

print("=" * 55)
print("ASFR MODEL VERIFICATION")
print("=" * 55)

# 1. Read metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 2. Verify checkpoint hash
sha256 = hashlib.sha256()

with open(MODEL_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha256.update(chunk)

actual_hash = sha256.hexdigest()

print("\nCheckpoint SHA-256:", actual_hash)
assert actual_hash == metadata["artifact_sha256"], "Checkpoint hash mismatch!"

# 3. Load model without compiling training configuration
print("\nLoading ASFR checkpoint...")
model = keras.models.load_model(MODEL_PATH, compile=False)

# 4. Inspect model
print("\nTensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Model name:", model.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameter count:", model.count_params())

# 5. Verify architecture
print("\nChecking expected ASFR layers...")

layer_names = [layer.name for layer in model.layers]

for name in ["asfr_fusion_relu", "asfr_spatial_gate"]:
    found = name in layer_names
    print(f"{name}: {'FOUND' if found else 'NOT FOUND'}")

# 6. Verify dimensions and parameters
assert model.input_shape == (None, 224, 224, 3)
assert model.output_shape == (None, 9)
assert model.count_params() == metadata["parameter_count"]

print("\nModel output classes:", len(metadata["class_names_in_label_order"]))
print("\nMODEL LOAD AND STRUCTURAL CHECK COMPLETE")
