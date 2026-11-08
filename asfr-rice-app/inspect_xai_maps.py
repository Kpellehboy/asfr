
import base64
import json
import numpy as np
from io import BytesIO
from PIL import Image

with open("test_outputs/analyze_response.json", "r", encoding="utf-8") as f:
    data = json.load(f)

maps = {
    "Grad-CAM heatmap": data["explanation"]["heatmap_png_base64"],
    "ASFR spatial gate": data["asfr_gate"]["gate_png_base64"],
}

for name, encoded in maps.items():
    raw = base64.b64decode(encoded)
    image = Image.open(BytesIO(raw)).convert("L")
    values = np.asarray(image, dtype=np.float32) / 255.0

    print(f"\n{name}")
    print("-" * 40)
    print("Dimensions:", image.size)
    print("Minimum:", float(values.min()))
    print("Maximum:", float(values.max()))
    print("Mean:", float(values.mean()))
    print("Standard deviation:", float(values.std()))

    for percentile in [1, 10, 25, 50, 75, 90, 99]:
        print(
            f"P{percentile}:",
            float(np.percentile(values, percentile))
        )
