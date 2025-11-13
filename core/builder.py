# core/builder.py
import yaml
import importlib
from typing import Any, Dict
from core import registry


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_from_config(config: Dict[str, Any], registry_obj) -> Any:
    """Generic builder using registry and kwargs."""
    name = config["name"]
    params = config.get("params", {})
    cls = registry_obj.get(name)
    return cls(**params)

def build_model(cfg: Dict[str, Any]):
    return build_from_config(cfg["model"], registry.MODEL_REGISTRY)

def build_dataset(cfg: Dict[str, Any]):
    return build_from_config(cfg["dataset"], registry.DATASET_REGISTRY)

def build_loss(cfg: Dict[str, Any]):
    return build_from_config(cfg["loss"], registry.LOSS_REGISTRY)

def build_optimizer(cfg: Dict[str, Any]):
    return build_from_config(cfg["optimizer"], registry.OPTIMIZER_REGISTRY)
