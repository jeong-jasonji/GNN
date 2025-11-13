import tensorflow as tf
from models.self_supervised.base_ssl import BaseSSLModel
from models.builders.model_builder import build_model_from_design
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("simclr")
class SimCLR(BaseSSLModel):
    """
    SimCLR (simple contrastive learning) implementation (TensorFlow 2.x)
    Requires model design file for backbone and projection head.
    """

    def build_model(self):
        design_path = self.cfg["model"]["params"]["design_path"]
        self.encoder = build_model_from_design(design_path)  # Backbone
        emb_dim = self.cfg["model"]["params"].get("embedding_dim", 128)
        self.projection_head = build_model_from_design(head_design_path)

    def forward(self, inputs, training=False):
        """Inputs: two augmented views of the same image (x_i, x_j)."""
        x_i, x_j = inputs
        z_i = self.projection_head(self.encoder(x_i, training=training))
        z_j = self.projection_head(self.encoder(x_j, training=training))
        return z_i, z_j

    def compute_loss(self, outputs, targets=None):
        z_i, z_j = outputs
        return self.compute_contrastive_loss(z_i, z_j)
