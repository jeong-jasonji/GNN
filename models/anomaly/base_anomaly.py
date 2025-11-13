import tensorflow as tf
from core.base_model import BaseModel

class BaseAnomalyModel(BaseModel):
    """
    Base class for anomaly detection (autoencoders, patch-based, etc.)
    """
    def compute_reconstruction_loss(self, x, x_hat):
        return tf.reduce_mean(tf.square(x - x_hat))
