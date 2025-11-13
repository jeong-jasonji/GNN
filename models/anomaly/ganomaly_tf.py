import tensorflow as tf
from models.anomaly.base_anomaly import BaseAnomalyModel
from models.builders.model_builder import build_model_from_design
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("ganomaly")
class GANomaly(BaseAnomalyModel):
    """
    GANomaly (Akçay et al., 2018)
    Encoder → Decoder → Encoder'
    Loss = reconstruction + feature consistency + adversarial
    """

    def build_model(self):
        enc_path = self.cfg["model"]["params"]["encoder_design"]
        dec_path = self.cfg["model"]["params"]["decoder_design"]
        self.encoder = build_model_from_design(enc_path)
        self.decoder = build_model_from_design(dec_path)
        self.encoder_recon = build_model_from_design(enc_path)  # E' for latent consistency
        self.discriminator = build_model_from_design(disc_path)

    def forward(self, x, training=True):
        z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        z_hat = self.encoder_recon(x_hat, training=training)
        return x_hat, z, z_hat

    def compute_loss(self, outputs, targets):
        x_hat, z, z_hat = outputs
        recon_loss = tf.reduce_mean(tf.square(targets - x_hat))
        latent_loss = tf.reduce_mean(tf.square(z - z_hat))
        adv_loss = tf.reduce_mean((self.discriminator(x_hat) - 1) ** 2)
        return recon_loss + 0.5 * latent_loss + 0.1 * adv_loss
