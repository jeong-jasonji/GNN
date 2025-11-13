import tensorflow as tf
from models.self_supervised.base_ssl import BaseSSLModel
from models.builders.model_builder import build_model_from_design
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("supcon")
class SupervisedContrastiveModel(BaseSSLModel):
    """
    Supervised Contrastive Learning (SupCon)
    - Reuses BaseSSLModel contrastive framework
    - Incorporates label-based mask for positives
    """

    def build_model(self):
        design_path = self.cfg["model"]["params"]["design_path"]
        self.encoder = build_model_from_design(design_path)
        emb_dim = self.cfg["model"]["params"].get("embedding_dim", 128)
        self.projection_head = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])

    def forward(self, x, training=False):
        z = self.encoder(x, training=training)
        z = self.projection_head(z, training=training)
        return tf.math.l2_normalize(z, axis=1)

    def compute_loss(self, features, labels):
        """Supervised contrastive loss (Khosla et al., 2020)."""
        temperature = self.temperature
        z = tf.math.l2_normalize(features, axis=1)
        similarity_matrix = tf.matmul(z, z, transpose_b=True) / temperature
        labels = tf.reshape(labels, [-1, 1])
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        logits_mask = 1.0 - tf.eye(tf.shape(z)[0])
        mask = mask * logits_mask

        exp_sim = tf.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))

        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)
        loss = -tf.reduce_mean(mean_log_prob_pos)
        return loss
