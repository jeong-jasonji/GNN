import itertools, copy, json, os
from core.builder import load_config
from training.base_trainer import BaseTrainer

def run_grid_search(config_path, param_grid):
    cfg_base = load_config(config_path)
    keys, values = zip(*param_grid.items())
    results = []

    for combo in itertools.product(*values):
        cfg = copy.deepcopy(cfg_base)
        for k, v in zip(keys, combo):
            d, key = cfg, k.split(".")
            for kk in key[:-1]:
                d = d[kk]
            d[key[-1]] = v

        trainer = BaseTrainer(cfg)
        hist = trainer.train()
        val = min(hist.get("val_loss", [999]))
        results.append({"params": dict(zip(keys, combo)), "val_loss": val})

    outdir = os.path.join("experiments", "hypersearch", "grid_results.json")
    with open(outdir, "w") as f:
        json.dump(results, f, indent=2)
    return results
