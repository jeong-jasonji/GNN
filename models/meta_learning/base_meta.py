import tensorflow as tf
from core.base_model import BaseModel
from abc import abstractmethod

class BaseMetaModel(BaseModel):
    """Generic base for meta-learning (e.g., MAML, ProtoNet)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_way = cfg["meta"].get("n_way", 5)
        self.k_shot = cfg["meta"].get("k_shot", 1)
        self.q_query = cfg["meta"].get("q_query", 15)

    @abstractmethod
    def meta_train_step(self, episodes):
        """Perform one meta-update."""
        pass

    def compute_episode_accuracy(self, preds, labels):
        return tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(preds, -1), tf.argmax(labels, -1)), tf.float32)
        )
