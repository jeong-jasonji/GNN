import tensorflow as tf
import gc, os

def clear_tf_memory(verbose=True):
    """Frees up TensorFlow and Python memory after a training run."""
    if verbose:
        print("[Memory] Clearing TensorFlow session and cache...")

    # clear keras session graph
    tf.keras.backend.clear_session()

    # run garbage collector
    gc.collect()

    # optional: clear GPU cache (TF2 automatically does this, but we add a check)
    try:
        physical_gpus = tf.config.list_physical_devices('GPU')
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if verbose:
            print(f"[Memory] {len(physical_gpus)} GPU(s) memory growth re-enabled.")
    except Exception as e:
        if verbose:
            print(f"[Memory] Warning while resetting GPU memory: {e}")

def clear_dataset_cache(dataset):
    """Deletes dataset iterator references and clears TF cache."""
    try:
        del dataset
        tf.data.experimental.clear_cardinality_cache()
        gc.collect()
        print("[Memory] Dataset cache cleared.")
    except Exception:
        pass
