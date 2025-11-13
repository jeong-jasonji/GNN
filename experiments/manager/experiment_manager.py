import os, json, datetime, subprocess
from core.builder import load_config
from training.base_trainer import BaseTrainer
from experiments.manager.environment_utils import capture_environment
from experiments.manager.tracker import ExperimentTracker

class ExperimentManager:
    """Unified launcher and logger for all experiment types."""

    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.exp_name = self.cfg["experiment"]["name"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("experiments", "runs", f"{self.exp_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.tracker = ExperimentTracker(self.run_dir)

        # capture environment & config snapshot
        capture_environment(self.run_dir)
        with open(os.path.join(self.run_dir, "config_snapshot.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)

    def run(self):
        print(f"[Manager] Starting experiment: {self.exp_name}")
        trainer = BaseTrainer(self.cfg)
        history = trainer.train()
        self.tracker.log_history(history)
        self.tracker.finalize()
        print(f"[Manager] Experiment completed. Results in {self.run_dir}")
        return self.run_dir

"""
supports parallel/distributed search:

python -m experiments.manager.experiment_manager --config configs/train_unet.yaml
python -m experiments.manager.experiment_manager --config configs/train_supcon.yaml

OR:

import concurrent.futures
configs = ["configs/train_unet.yaml", "configs/train_supcon.yaml"]
with concurrent.futures.ProcessPoolExecutor() as ex:
    ex.map(lambda p: ExperimentManager(p).run(), configs)
"""