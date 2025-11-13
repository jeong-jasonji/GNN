import os, json, pandas as pd

def summarize_experiments(base_dir="experiments/runs"):
    records = []
    for run in os.listdir(base_dir):
        hist_file = os.path.join(base_dir, run, "training_history.json")
        if os.path.exists(hist_file):
            data = json.load(open(hist_file))
            records.append({
                "run": run,
                "final_loss": float(data.get("loss", [None])[-1] or 0),
                "final_val_loss": float(data.get("val_loss", [None])[-1] or 0)
            })
    df = pd.DataFrame(records)
    print(df.sort_values("final_val_loss").head())
    return df
