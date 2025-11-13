import tensorflow as tf

def log_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        details = tf.config.experimental.get_memory_info('GPU:0')
        used = details['current'] / (1024**3)
        peak = details['peak'] / (1024**3)
        print(f"[GPU] Memory used: {used:.2f} GB | Peak: {peak:.2f} GB")
