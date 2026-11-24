import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from app.inference import InferenceEngine
from app.xai import GradCAMExplainer

MODEL_PATH = Path("model_artifacts/best_model.keras")
CLASS_PATH = Path("model_artifacts/class_names.json")
METADATA_PATH = Path("model_artifacts/model_metadata.json")
IMAGE_PATH = Path("test_images/20231006_164208.jpg")

engine = InferenceEngine(
    model_path=MODEL_PATH,
    class_names_path=CLASS_PATH,
    metadata_path=METADATA_PATH,
)
engine.load()

explainer = GradCAMExplainer(engine)

image_bytes = IMAGE_PATH.read_bytes()
_, batch = explainer._prepare(image_bytes)

gate_batch = explainer.gate_model(
    batch,
    training=False,
).numpy()

gate = gate_batch[0, ..., 0]

print("\nNATIVE ASFR GATE DISTRIBUTION")
print("=" * 45)
print("Shape:", gate.shape)
print("Minimum:", float(gate.min()))
print("Maximum:", float(gate.max()))
print("Mean:", float(gate.mean()))
print("Standard deviation:", float(gate.std()))

for percentile in [0, 1, 10, 25, 50, 75, 90, 99, 100]:
    print(
        f"P{percentile}:",
        float(np.percentile(gate, percentile))
    )

print("\nNative gate values:")
np.set_printoptions(precision=4, suppress=True)
print(gate)
