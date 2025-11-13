import tensorflow as tf
import numpy as np
from evaluation.sanity_checks.visualization_utils import plot_reconstructions
from evaluation.sanity_checks.logger import DiagnosticsLogger

def reconstruction_diagnostics(model, dataset, log_dir=None, num_samples=5, step=0):
    """Visualize reconstructions, compute losses, and log them."""
    x_batch, _ = next(iter(dataset))
    x_hat = model(x_batch, training=False)
    losses = tf.reduce_mean(tf.square(x_batch - x_hat), axis=[1,2,3])
    mean_loss = tf.reduce_mean(losses).numpy()
    print(f"[AnomalyCheck] Mean recon loss: {mean_loss:.4f}")

    if log_dir:
        logger = DiagnosticsLogger(log_dir)
        logger.log_scalar("anomaly/reconstruction_loss", mean_loss, step)
        fig = plot_reconstructions(x_batch.numpy(), x_hat.numpy(), num_samples=num_samples)
        logger.log_figure("anomaly/reconstruction_examples", fig, step)
        logger.close()

def gan_feature_diagnostics(model, dataset):
    """Compute discriminator feature consistency for GANomaly / f-AnoGAN."""
    x_batch, _ = next(iter(dataset))
    outputs = model.forward(x_batch, training=False)
    if len(outputs) >= 3:
        _, f_real, f_fake, _ = outputs[:4]
        feat_diff = tf.reduce_mean(tf.abs(f_real - f_fake))
        print(f"[GANCheck] Feature consistency = {feat_diff.numpy():.4f}")
