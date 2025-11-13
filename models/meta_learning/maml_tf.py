import tensorflow as tf
from models.meta_learning.base_meta import BaseMetaModel
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("maml")
class MAML(BaseMetaModel):
    """Simplified first-order MAML for TensorFlow."""

    def meta_train_step(self, episodes):
        s_x, s_y, q_x, q_y = episodes
        inner_lr = self.cfg["meta"].get("inner_lr", 0.01)
        loss_fn  = tf.keras.losses.get(self.cfg["loss"]["name"])

        # clone weights
        weights = self.encoder.get_weights()
        with tf.GradientTape() as tape:
            preds = self.encoder(s_x, training=True)
            inner_loss = loss_fn(s_y, preds)
        grads = tape.gradient(inner_loss, self.encoder.trainable_variables)
        updated_weights = [w - inner_lr * g for w, g in zip(weights, grads)]

        # fast-adapted forward
        self.encoder.set_weights(updated_weights)
        preds_q = self.encoder(q_x, training=True)
        meta_loss = loss_fn(q_y, preds_q)

        # restore weights after meta step
        self.encoder.set_weights(weights)
        return meta_loss
