import json, os, tensorflow as tf

class ExperimentTracker:
    """Handles local + TensorBoard tracking for experiments."""
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, "logs")
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.history_path = os.path.join(run_dir, "training_history.json")

    def log_history(self, history_dict):
        with open(self.history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        # write final metrics to TensorBoard
        with self.writer.as_default():
            for key, values in history_dict.items():
                for i, v in enumerate(values):
                    tf.summary.scalar(f"train/{key}", v, step=i)
        self.writer.flush()

    def finalize(self):
        self.writer.close()
        print("[Tracker] Training metrics logged to TensorBoard.")
