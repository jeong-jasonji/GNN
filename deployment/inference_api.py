import tensorflow as tf
import numpy as np
import json, os
from core.builder import load_config, build_model

class InferenceAPI:
    """Unified prediction interface for supervised, SSL, anomaly, or meta models."""
    def __init__(self, config_path, weights_path=None):
        self.cfg = load_config(config_path)
        self.model = build_model(self.cfg)
        if weights_path:
            print(f"[Inference] Loading weights from {weights_path}")
            self.model.load_weights(weights_path)
        self.model(tf.random.normal((1, *self.cfg["model"]["params"]["input_shape"])))  # build

    def predict(self, inputs, return_features=False):
        preds = self.model(inputs, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if return_features and hasattr(self.model, "encoder"):
            feats = self.model.encoder(inputs, training=False)
            return preds, feats
        return preds

    def classify_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=self.cfg["model"]["params"]["input_shape"][:2])
        x = tf.expand_dims(tf.keras.utils.img_to_array(img) / 255.0, 0)
        preds = self.predict(x)
        return np.array(preds[0])

    def anomaly_score(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=self.cfg["model"]["params"]["input_shape"][:2])
        x = tf.expand_dims(tf.keras.utils.img_to_array(img) / 255.0, 0)
        x_hat = self.model(x, training=False)
        loss = tf.reduce_mean(tf.square(x - x_hat)).numpy()
        return float(loss)

"""
Example Usage
from deployment.inference_api import InferenceAPI

api = InferenceAPI("configs/train_lightcnn_design.yaml",
                   "experiments/runs/lightcnn_design_test/final_model/variables/variables")

preds = api.classify_image("sample_defect.jpg")
print("Predictions:", preds)

score = api.anomaly_score("sample_defect.jpg")
print("Anomaly score:", score)
"""