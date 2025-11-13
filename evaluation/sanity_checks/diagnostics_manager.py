import os
import tensorflow as tf
from evaluation.sanity_checks import (
    generic, ssl_checks, anomaly_checks, meta_checks
)
from evaluation.sanity_checks.logger import DiagnosticsLogger
from data.meta_tasks.episodic_loader_tf import EpisodicLoader

class DiagnosticsManager:
    """
    Unified diagnostics interface for models.
    Can run generic, SSL, or anomaly checks and log results.
    """

    def __init__(self, model, val_ds, log_dir, cfg):
        self.model = model
        self.val_ds = val_ds
        self.log_dir = log_dir
        self.cfg = cfg
        os.makedirs(log_dir, exist_ok=True)
        self.logger = DiagnosticsLogger(log_dir)

    def run_all(self, step=0):
        print(f"\n[Diagnostics] Running full diagnostics at step {step}...")
        model_name = self.model.__class__.__name__.lower()

        # --- Generic checks ---
        try:
            mean_grad, max_grad = generic.gradient_health_check(
                self.model, tf.keras.losses.get(self.cfg["loss"]["name"])
            )
            self.logger.log_scalar("generic/mean_gradient_norm", mean_grad, step)
            self.logger.log_scalar("generic/max_gradient_norm", max_grad, step)
        except Exception as e:
            print(f"[Diagnostics] Generic check failed: {e}")

        # --- Forward pass check ---
        try:
            shape = generic.check_forward_pass(
                self.model, input_shape=tuple(self.cfg["model"]["params"]["input_shape"])
            )
            print(f"[Diagnostics] Forward pass shape OK: {shape}")
        except Exception as e:
            print(f"[Diagnostics] Forward pass failed: {e}")

        # --- Overfit test (optional small run) ---
        if self.cfg.get("run_overfit_test", False):
            try:
                loss = generic.overfit_one_batch(
                    self.model,
                    self.val_ds,
                    tf.keras.optimizers.get(self.cfg["optimizer"]["name"]),
                    tf.keras.losses.get(self.cfg["loss"]["name"]),
                    steps=10,
                )
                self.logger.log_scalar("generic/overfit_one_batch_loss", loss, step)
            except Exception as e:
                print(f"[Diagnostics] Overfit test skipped: {e}")

        # --- SSL or anomaly-specific diagnostics ---
        try:
            if "supcon" in model_name or "simclr" in model_name:
                ssl_checks.embedding_diagnostics(
                    self.model, self.val_ds, log_dir=self.log_dir, step=step
                )
            elif any(k in model_name for k in ["autoencoder", "ganomaly", "fanogan"]):
                anomaly_checks.reconstruction_diagnostics(
                    self.model, self.val_ds, log_dir=self.log_dir, step=step
                )
        except Exception as e:
            print(f"[Diagnostics] Model-specific diagnostics skipped: {e}")

        # --- Meta-learning diagnostics ---
        try:
            if "maml" in model_name or "protonet" in model_name or "reptile" in model_name:
                episodic_loader = EpisodicLoader(self.val_ds)
                meta_checks.meta_diagnostics(self.model, episodic_loader, log_dir=self.log_dir, step=step)
        except Exception as e:
            print(f"[Diagnostics] Meta-learning diagnostics skipped: {e}")

        print(f"[Diagnostics] Completed diagnostics at step {step}.")
        self.logger.close()


"""
Optional testing of SSL encoders on meta-learning:
ssl_encoder = tf.keras.models.load_model("experiments/runs/supcon_defects/final_model")
proto_model.encoder.set_weights(ssl_encoder.get_weights())
meta_checks.meta_diagnostics(proto_model, episodic_loader, log_dir=diag_dir, step=0)
"""