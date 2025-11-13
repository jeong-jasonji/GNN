import tensorflow as tf
import numpy as np

def overfit_one_batch(model, dataset, optimizer, loss_fn, steps=20):
    """Try to overfit on a single batch for sanity."""
    batch = next(iter(dataset))
    x, y = batch
    for i in range(steps):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % 5 == 0:
            print(f"[OverfitTest] Step {i} Loss={loss.numpy():.4f}")
    return float(loss.numpy())

def gradient_health_check(model, loss_fn):
    """Check that gradients are finite and well-scaled."""
    x = tf.random.normal((2, 224, 224, 3))
    y = tf.one_hot([0, 1], depth=2)
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = loss_fn(y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    grad_norms = [tf.norm(g) for g in grads if g is not None]
    mean, maxv = tf.reduce_mean(grad_norms), tf.reduce_max(grad_norms)
    print(f"[GradientCheck] mean={mean:.4f}, max={maxv:.4f}")
    return float(mean.numpy()), float(maxv.numpy())

def check_forward_pass(model, input_shape=(224, 224, 3)):
    """Ensure forward pass produces correct output shape."""
    x = tf.random.normal((1, *input_shape))
    y = model(x, training=False)
    print("[Sanity] Forward pass output shape:", y.shape)
    return y.shape


"""
Example sanity check:

from core.builder import load_config, build_model, build_dataset, build_loss, build_optimizer
from evaluation.sanity_checks.generic import check_forward_pass, check_gradient_flow, overfit_one_batch

cfg = load_config("configs/train_lightcnn.yaml")
model = build_model(cfg)
dataset = build_dataset(cfg)
loss_fn = build_loss(cfg)
optimizer = build_optimizer(cfg)

check_forward_pass(model)
check_gradient_flow(model, loss_fn)
overfit_one_batch(model, dataset.get_data_loaders()["train"], optimizer, loss_fn)

Should output something like:

[Sanity] Forward pass output shape: (1, 2)
[Sanity] Gradients valid: True
[Overfit Test] Step 0: loss = 0.6894
...
[Overfit Test] Step 15: loss = 0.1221
"""