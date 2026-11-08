
import json
import numpy as np
import tensorflow as tf
import keras

from PIL import Image
from pathlib import Path

MODEL_PATH = Path("model_artifacts/best_model.keras")
CLASS_PATH = Path("model_artifacts/class_names.json")
TEST_IMAGE_DIR = Path("test_images")

# Load class mapping
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)

class_names = class_data["class_names"]

# Load model
model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Find test images only
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_files = [
    p for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

if not image_files:
    raise FileNotFoundError(
        "No image found in test_images. Add a rice-leaf image first."
    )

image_path = image_files[0]
print("Testing image:", image_path)

# Preprocess image
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

image_array = np.asarray(image, dtype=np.float32)

image_array = tf.keras.applications.resnet_v2.preprocess_input(
    image_array
)

image_array = np.expand_dims(image_array, axis=0)

# Run inference
predictions = model.predict(image_array, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
confidence = float(predictions[predicted_index])

print("\nPREDICTION RESULTS")
print("------------------")
print("Predicted class:", predicted_class)
print(f"Confidence: {confidence:.4f}")
print(f"Confidence percentage: {confidence * 100:.2f}%")

print("\nALL CLASS PROBABILITIES")
print("----------------------")

for name, probability in zip(class_names, predictions):
    print(f"{name}: {probability:.4f}")

# Validate output
assert len(predictions) == len(class_names) == 9
assert np.all(np.isfinite(predictions))
assert np.all(predictions >= 0)
assert np.all(predictions <= 1)
assert np.isclose(np.sum(predictions), 1.0, atol=1e-4)

print("\nPREDICTION TEST PASSED")
