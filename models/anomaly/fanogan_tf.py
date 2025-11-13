import tensorflow as tf
from models.anomaly.base_anomaly import BaseAnomalyModel
from models.builders.model_builder import build_model_from_design
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("fanogan")
class fAnoGAN(BaseAnomalyModel):
    """
    f-AnoGAN (Schlegl et al., 2019)
    Trains GAN on normal data, then learns an encoder to map images to latent space.
    """

    def build_model(self):
        gen_path = self.cfg["model"]["params"]["generator_design"]
        disc_path = self.cfg["model"]["params"]["discriminator_design"]
        enc_path = self.cfg["model"]["params"]["encoder_design"]

        self.generator = build_model_from_design(gen_path)
        self.discriminator = build_model_from_design(disc_path)
        self.encoder = build_model_from_design(enc_path)

    def forward(self, x, training=False):
        """Compute reconstruction and feature differences."""
        z = self.encoder(x, training=training)
        x_hat = self.generator(z, training=training)
        f_real = self.discriminator(x, training=training)
        f_fake = self.discriminator(x_hat, training=training)
        return x_hat, f_real, f_fake, z

    def compute_loss(self, outputs, targets):
        x_hat, f_real, f_fake, z = outputs
        recon_loss = tf.reduce_mean(tf.abs(targets - x_hat))
        feature_loss = tf.reduce_mean(tf.abs(f_real - f_fake))
        return recon_loss + 0.1 * feature_loss
