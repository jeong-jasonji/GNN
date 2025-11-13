import optuna, os, json
from core.builder import load_config
from training.base_trainer import BaseTrainer
from core.utils.memory_utils import clear_tf_memory, clear_dataset_cache

def objective(trial, base_config_path):
    cfg = load_config(base_config_path)

    # sample hyperparameters
    cfg["optimizer"]["params"]["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    cfg["trainer"]["epochs"] = trial.suggest_int("epochs", 20, 60)
    cfg["model"]["params"]["filters"] = trial.suggest_categorical("filters", [
        [16, 32, 64], [32, 64, 128], [64, 128, 256]
    ])
    # optional architecture parameters
    cfg["model"]["params"]["design_path"] = trial.suggest_categorical(
        "design_path",
        ["configs/models/light_cnn_design.yaml", "configs/models/residual_design.yaml"]
    )

    trainer = BaseTrainer(cfg)
    history = trainer.train()
    best_val = min(history.get("val_loss", [999]))
    clear_tf_memory(verbose=False) # clear memory
    clear_dataset_cache(self.train_ds)
    clear_dataset_cache(self.val_ds)
    return best_val

def run_optuna_search(base_config_path, n_trials=10, study_name="hparam_search"):
    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(lambda t: objective(t, base_config_path), n_trials=n_trials)

    results_dir = os.path.join("experiments", "hypersearch", study_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "study_results.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[Optuna] Best params: {study.best_params}")
    return study
