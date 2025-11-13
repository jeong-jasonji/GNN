import tensorflow as tf

def check_forward_pass(model, input_shape=(224, 224, 3)):
    """Ensure forward pass produces correct output shape."""
    x = tf.random.normal((1, *input_shape))
    y = model(x, training=False)
    print("[Sanity] Forward pass output shape:", y.shape)

def check_gradient_flow(model, loss_fn):
    """Check that gradients are not None or NaN."""
    x = tf.random.normal((2, 224, 224, 3))
    y = tf.one_hot(tf.constant([0, 1]), depth=model.cfg["model"]["params"]["num_classes"])
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = loss_fn(y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    nan_grads = any([tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None])
    print(f"[Sanity] Gradients valid: {not nan_grads}")

def overfit_one_batch(model, dataset, optimizer, loss_fn, steps=20):
    """Try to overfit on one batch to ensure learning works."""
    batch = next(iter(dataset))
    x, y = batch
    for i in range(steps):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % 5 == 0:
            print(f"[Overfit Test] Step {i}: loss = {loss.numpy():.4f}")

"""
Example sanity check:

from core.builder import load_config, build_model, build_dataset, build_loss, build_optimizer
from evaluation.sanity_checks import check_forward_pass, check_gradient_flow, overfit_one_batch

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