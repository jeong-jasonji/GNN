import tensorflow as tf
from abc import ABC, abstractmethod
import os

class BaseDataset(ABC):
    """Abstract dataset interface supporting train/val/test splits."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg["dataset"]["params"]["data_dir"]
        self.batch_size = cfg["dataset"]["params"]["batch_size"]
        self.augment = cfg["dataset"]["params"].get("augmentations", True)
        self.prepare_data()

    @abstractmethod
    def prepare_data(self):
        """Any one-time preprocessing or dataset download."""
        pass

    @abstractmethod
    def load_dataset(self, split: str):
        """Return a tf.data.Dataset object for the given split."""
        pass

    def get_data_loaders(self):
        """Convenience wrapper."""
        train_ds = self.load_dataset("train")
        val_ds = self.load_dataset("val")
        return {"train": train_ds, "val": val_ds}
