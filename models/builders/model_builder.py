import tensorflow as tf
import yaml, json
from models.backbones.model_components import ConvBlock, ResidualBlock, DenseBlock

# Map layer names to classes
BLOCK_REGISTRY = {
    "ConvBlock": ConvBlock,
    "ResidualBlock": ResidualBlock,
    "Dense": tf.keras.layers.Dense,
    "DenseBlock": DenseBlock,
    "Flatten": tf.keras.layers.Flatten,
    "GlobalAvgPool": tf.keras.layers.GlobalAveragePooling2D,
}

def load_design_file(path: str):
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported design file format: {path}")

def build_model_from_design(design_path: str):
    """Builds a tf.keras.Model based on a YAML/JSON design file."""
    design = load_design_file(design_path)
    layers_config = design["model_design"]["layers"]
    input_shape = tuple(design["model_design"]["input_shape"])
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for layer_cfg in layers_config:
        layer_type = layer_cfg.pop("type")
        if layer_type not in BLOCK_REGISTRY:
            raise ValueError(f"Unknown block type: {layer_type}")
        layer_class = BLOCK_REGISTRY[layer_type]
        layer = layer_class(**layer_cfg)
        x = layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="dynamic_model")
    return model
