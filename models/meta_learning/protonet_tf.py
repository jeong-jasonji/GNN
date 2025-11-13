import tensorflow as tf
from models.meta_learning.base_meta import BaseMetaModel
from models.builders.model_builder import build_model_from_design
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("protonet")
class ProtoNet(BaseMetaModel):
    """Prototypical Network (Snell et al., 2017)."""

    def build_model(self):
        encoder_path = self.cfg["model"]["params"]["encoder_design"]
        self.encoder = build_model_from_design(encoder_path)

    def forward(self, x, training=False):
        z = self.encoder(x, training=training)
        return tf.math.l2_normalize(z, axis=-1)

    def meta_train_step(self, episodes):
        """Each episode = (support_images, support_labels, query_images, query_labels)."""
        s_x, s_y, q_x, q_y = episodes
        z_support = self.forward(s_x, training=True)
        z_query   = self.forward(q_x, training=True)

        classes = tf.unique(s_y)[0]
        prototypes = [tf.reduce_mean(tf.boolean_mask(z_support, s_y == c), axis=0) for c in classes]
        prototypes = tf.stack(prototypes)

        # compute distances and probabilities
        dists = tf.norm(tf.expand_dims(z_query, 1) - tf.expand_dims(prototypes, 0), axis=-1)
        probs = tf.nn.softmax(-dists, axis=1)
        y_onehot = tf.one_hot(s_y.numpy().astype(int), depth=len(classes))
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_onehot, probs))
        acc  = self.compute_episode_accuracy(probs, y_onehot)
        return loss, acc
