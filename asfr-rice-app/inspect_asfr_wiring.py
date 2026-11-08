
import tensorflow as tf

MODEL_PATH = "model_artifacts/best_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

TARGET_LAYERS = [
    "asfr_gate_input",
    "asfr_spatial_gate",
    "asfr_gated_deep",
    "asfr_asymmetric_add",
    "asfr_fusion_relu",
]


def describe_tensor(tensor):
    history = getattr(tensor, "_keras_history", None)

    if history is not None:
        source_layer = history.operation.name
    else:
        source_layer = "Unknown"

    return {
        "source_layer": source_layer,
        "shape": tuple(tensor.shape),
    }


print("\nASFR COMPUTATIONAL GRAPH")
print("=" * 65)

for layer_name in TARGET_LAYERS:
    layer = model.get_layer(layer_name)

    print(f"\nLAYER: {layer.name}")
    print("TYPE:", layer.__class__.__name__)

    inputs = layer.input
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    print("INPUTS:")
    for index, tensor in enumerate(inputs):
        print(f"  [{index}] {describe_tensor(tensor)}")

    outputs = layer.output
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    print("OUTPUTS:")
    for index, tensor in enumerate(outputs):
        print(f"  [{index}] {describe_tensor(tensor)}")
