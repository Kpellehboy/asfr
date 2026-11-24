
import base64
import json
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:8000/analyze"
IMAGE_PATH = Path("test_images/20231006_164208.jpg")
OUTPUT_DIR = Path("test_outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

print("Sending image to ASFR analysis endpoint...")

with IMAGE_PATH.open("rb") as image:
    response = requests.post(
        API_URL,
        files={
            "file": (
                IMAGE_PATH.name,
                image,
                "image/jpeg"
            )
        },
        timeout=180
    )

print("\nHTTP STATUS:", response.status_code)
response.raise_for_status()

result = response.json()

# Save complete JSON response
with open(OUTPUT_DIR / "analyze_response.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

prediction = result["prediction"]
explanation = result["explanation"]
gate = result["asfr_gate"]

print("\nPREDICTION")
print("----------")
print("Class:", prediction["predicted_class"])
print("Confidence:", prediction["confidence"])
print("Model:", prediction["model_version"])

print("\nGRAD-CAM EXPLANATION")
print("--------------------")
print("Explained class:", explanation["explained_class"])
print("Method:", explanation["method"])
print("Target layer:", explanation["target_layer"])
print("Warning:", explanation["warning"])

print("\nASFR SPATIAL GATE")
print("-----------------")
print("Method:", gate["method"])
print("Target layer:", gate["target_layer"])
print("Warning:", gate["warning"])

# Decode and validate PNG outputs
outputs = {
    "gradcam_heatmap.png": explanation["heatmap_png_base64"],
    "gradcam_overlay.png": explanation["overlay_png_base64"],
    "asfr_gate.png": gate["gate_png_base64"],
}

for filename, encoded_image in outputs.items():
    image_bytes = base64.b64decode(encoded_image, validate=True)

    # Verify that the data is a valid PNG image
    image = Image.open(BytesIO(image_bytes))
    image.load()

    if image.format != "PNG":
        raise ValueError(f"{filename} is not a PNG image")

    image.save(OUTPUT_DIR / filename)

    print(
        f"\nSaved: {filename} "
        f"| Dimensions: {image.width}x{image.height}"
    )

# Basic consistency checks
assert prediction["predicted_class"] == explanation["explained_class"]
assert prediction["predicted_class_index"] == explanation["explained_class_index"]
assert prediction["model_version"] == explanation["model_version"]
assert prediction["model_version"] == gate["model_version"]
assert result["request_id"] == prediction["request_id"]
assert result["request_id"] == explanation["request_id"]
assert result["request_id"] == gate["request_id"]

print("\nASFR ANALYSIS TEST PASSED")
print("Output directory:", OUTPUT_DIR.resolve())
