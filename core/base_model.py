import tensorflow as tf
from abc import ABC, abstractmethod

class BaseModel(tf.keras.Model, ABC):
    """Abstract base model for all CV tasks (supervised or SSL)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_model()

    @abstractmethod
    def build_model(self):
        """Define model layers here."""
        pass

    @abstractmethod
    def forward(self, inputs, training=False):
        """Forward pass."""
        pass

    def call(self, inputs, training=False):
        return self.forward(inputs, training)

    def compute_loss(self, outputs, targets):
        """Default supervised loss. Override for SSL/meta."""
        loss_fn = tf.keras.losses.get(self.cfg["loss"]["name"])
        return loss_fn(targets, outputs)

    def training_step(self, data):
        """Handles one training step (for custom loops)."""
        x, y = data
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = self.compute_loss(preds, y)
        grads = tape.gradient(loss, self.trainable_variables)
        return loss, grads

    def compute_diagnostics(self, dataset, log_dir=None, step=0):
        from evaluation.sanity_checks import ssl_checks, anomaly_checks
        model_name = self.__class__.__name__.lower()

        if "supcon" in model_name or "simclr" in model_name:
            ssl_checks.embedding_diagnostics(self, dataset, log_dir=log_dir, step=step)
        elif any(k in model_name for k in ["autoencoder", "ganomaly", "fanogan"]):
            anomaly_checks.reconstruction_diagnostics(self, dataset, log_dir=log_dir, step=step)
