# core/registry.py
from typing import Dict, Type, Any, Callable

class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, key: str) -> Callable:
        """Decorator to register a class or function under a key."""
        def decorator(obj: Any):
            if key in self._registry:
                raise KeyError(f"{key} already registered in {self._name}")
            self._registry[key] = obj
            return obj
        return decorator

    def get(self, key: str) -> Any:
        if key not in self._registry:
            raise KeyError(f"{key} not found in {self._name} registry")
        return self._registry[key]

    def list(self):
        return list(self._registry.keys())


# Instantiate global registries
MODEL_REGISTRY = Registry("Model")
DATASET_REGISTRY = Registry("Dataset")
LOSS_REGISTRY = Registry("Loss")
OPTIMIZER_REGISTRY = Registry("Optimizer")
SCHEDULER_REGISTRY = Registry("Scheduler")