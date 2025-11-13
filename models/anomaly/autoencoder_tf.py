import tensorflow as tf
from models.anomaly.base_anomaly import BaseAnomalyModel
from models.builders.model_builder import build_model_from_design
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("autoencoder")
class AutoEncoder(BaseAnomalyModel):
    """
    Generic autoencoder using design files for encoder and decoder parts.
    """

    def build_model(self):
        enc_path = self.cfg["model"]["params"]["encoder_design"]
        dec_path = self.cfg["model"]["params"]["decoder_design"]
        self.encoder = build_model_from_design(enc_path)
        self.decoder = build_model_from_design(dec_path)

    def forward(self, x, training=False):
        latent = self.encoder(x, training=training)
        reconstructed = self.decoder(latent, training=training)
        return reconstructed

    def compute_loss(self, outputs, targets):
        return self.compute_reconstruction_loss(targets, outputs)
