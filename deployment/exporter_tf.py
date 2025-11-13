import os, datetime, tensorflow as tf
from deployment.compression_utils import prune_model, quantize_model, cluster_model, save_tflite_model

def export_saved_model(model, export_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"version_{timestamp}")
    tf.saved_model.save(model, export_path)
    print(f"[Export] SavedModel exported to {export_path}")
    return export_path

def export_onnx(model, export_dir, sample_input):
    try:
        import tf2onnx
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(export_dir, f"version_{timestamp}.onnx")
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[sample_input])
        with open(export_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"[Export] ONNX model exported to {export_path}")
        return export_path
    except ImportError:
        print("[Export] tf2onnx not installed, skipping ONNX export.")

def export_compressed(model, export_dir, representative_ds=None):
    print("[Export] Creating compressed model variants...")
    # Pruned
    pruned = prune_model(model, sparsity=0.5)
    pruned.save(os.path.join(export_dir, "pruned_model"))
    # Clustered
    clustered = cluster_model(model, num_clusters=8)
    clustered.save(os.path.join(export_dir, "clustered_model"))
    # Quantized TFLite
    tflite = quantize_model(model, representative_ds)
    save_tflite_model(tflite, export_dir)

# export_compressed(self.model, self.ckpt_dir, representative_ds=self.val_ds)