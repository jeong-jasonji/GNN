import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os, datetime

def prune_model(model, sparsity=0.5):
    """Applies global weight pruning."""
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruned_model = prune_low_magnitude(model, 
        pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
            sparsity, begin_step=0))
    return pruned_model

def quantize_model(model, representative_ds=None):
    """Converts to INT8 TFLite model (post-training quantization)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_ds is not None:
        def representative_data_gen():
            for x, _ in representative_ds.take(100):
                yield [x]
        converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()
    return tflite_model

def cluster_model(model, num_clusters=8):
    """Weight clustering to reduce redundancy."""
    cluster_weights = tfmot.clustering.keras.cluster_weights
    clustered = cluster_weights(model,
        number_of_clusters=num_clusters,
        cluster_centroids_init=tfmot.clustering.keras.CentroidInitialization.DENSITY_BASED)
    return clustered

def save_tflite_model(tflite_model, export_dir, name="model"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(export_dir, f"{name}_{ts}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"[Compression] TFLite model saved to {path}")
    return path
