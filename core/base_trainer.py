import tensorflow as tf
import numpy as np
import shutil
import os
from core.utils.seed_utils import set_global_seed
from core.builder import build_model, build_dataset, build_loss, build_optimizer
from evaluation.sanity_checks.diagnostics_manager import DiagnosticsManager
from deployment.exporter_tf import export_saved_model
from deployment.version_manager import register_model_version
from core.utils.memory_utils import clear_tf_memory, clear_dataset_cache

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg["experiment"].get("seed", 42)
        set_global_seed(self.seed)
        self.strategy = tf.distribute.get_strategy()
        self.setup()

    def setup(self):
        print("[Trainer] Building model, dataset, optimizer, and loss...")
        self.model = build_model(self.cfg)
        self.dataset = build_dataset(self.cfg)
        self.loss_fn = build_loss(self.cfg)
        self.optimizer = build_optimizer(self.cfg)

        self.train_ds = self.dataset.get_data_loaders()["train"]
        self.val_ds = self.dataset.get_data_loaders()["val"]

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["accuracy"]
        )

        self.ckpt_dir = os.path.join("experiments", "runs", self.cfg["experiment"]["name"])
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        # --- Step 0: set configurations and diagnostic paths ---
        config_path = os.path.join(self.ckpt_dir, "config_snapshot.yaml")
        shutil.copy("configs/train_default.yaml", config_path)

        diag_dir = os.path.join(self.ckpt_dir, "diagnostics")
        diagnostics = DiagnosticsManager(self.model, self.val_ds, diag_dir, self.cfg)

        # --- Step 1: Pre-training diagnostics ---
        if self.cfg.get("diagnostics", True):
            diagnostics.run_all(step=0)

        # --- Step 2: Custom TensorBoard callback with periodic diagnostics ---
        class PeriodicDiagnosticsCallback(tf.keras.callbacks.Callback):
            def __init__(self, diagnostics, interval=5):
                super().__init__()
                self.diagnostics = diagnostics
                self.interval = interval

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.interval == 0:
                    self.diagnostics.run_all(step=epoch + 1)

        print(f"[Trainer] Starting training for {self.cfg['trainer']['epochs']} epochs...")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg["trainer"]["epochs"],
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.ckpt_dir, "logs")),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.ckpt_dir, "best_model.h5"),
                    monitor="val_loss",
                    save_best_only=True
                ),
                PeriodicDiagnosticsCallback(diagnostics, interval=self.cfg["trainer"].get("diagnostics_interval", 5))
            ]
        )

        # --- Step 3: Post-training diagnostics ---
        if self.cfg.get("diagnostics", True):
            diagnostics.run_all(step=self.cfg["trainer"]["epochs"])

        # --- Step 4: Save final model ---
        self.model.save(os.path.join(self.ckpt_dir, "final_model"))
        print(f"[Trainer] Training complete. Model + diagnostics saved to {self.ckpt_dir}")

        # --- Step 5: Register and export the model after training.
        export_path = export_saved_model(self.model, self.ckpt_dir)
        register_model_version(self.ckpt_dir, export_path, metrics={"val_loss": float(min(history["val_loss"]))})
        
        if cfg.get("export", {}).get("compress", False):
            export_compressed(self.model, self.ckpt_dir, self.val_ds)

        # --- Step 6: clear memory
        clear_tf_memory(verbose=True)
        clear_dataset_cache(self.train_ds)
        clear_dataset_cache(self.val_ds)
        return history.history
