import os, tensorflow as tf

def report_model_size(model_path):
    size = os.path.getsize(model_path) / (1024**2)
    print(f"[Report] Model size: {size:.2f} MB")
    return size

def benchmark_tflite(tflite_path, sample):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"[Report] Inference output shape: {output.shape}")
