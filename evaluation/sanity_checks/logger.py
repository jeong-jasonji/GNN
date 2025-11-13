import os
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

class DiagnosticsLogger:
    """Handles TensorBoard + Matplotlib logging for diagnostics."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step=0):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def log_figure(self, tag, fig, step=0):
        """Save Matplotlib figure and log to TensorBoard."""
        fig_path = os.path.join(self.log_dir, f"{tag.replace('/', '_')}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        with self.writer.as_default():
            img = tf.io.read_file(fig_path)
            img = tf.image.decode_png(img, channels=4)
            img = tf.expand_dims(img, 0)
            tf.summary.image(tag, img, step=step)
            self.writer.flush()
        return fig_path

    def close(self):
        self.writer.close()
