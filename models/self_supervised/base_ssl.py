import tensorflow as tf
from core.base_model import BaseModel

class BaseSSLModel(BaseModel):
    """
    Base class for self-supervised models (SimCLR, BYOL, etc.).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.temperature = cfg["model"]["params"].get("temperature", 0.1)
        self.embedding_dim = cfg["model"]["params"].get("embedding_dim", 128)

    def compute_contrastive_loss(self, z_i, z_j):
        """NT-Xent loss for SimCLR-style models."""
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)

        logits = tf.matmul(z_i, z_j, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(logits)[0])
        loss_i = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_j = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return tf.reduce_mean(loss_i + loss_j) / 2
