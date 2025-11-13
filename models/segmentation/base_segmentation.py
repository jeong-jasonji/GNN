import tensorflow as tf
from core.base_model import BaseModel

class BaseSegmentationModel(BaseModel):
    """Extended base class for segmentation networks with multiple loss and metric options."""

    # ---------- LOSSES ----------
    def compute_loss(self, preds, masks):
        loss_name = self.cfg["loss"]["name"].lower()
        if loss_name == "dice":
            return self.dice_loss(masks, preds)
        elif loss_name == "bce":
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(masks, preds))
        elif loss_name == "cce":
            return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(masks, preds))
        elif loss_name == "focal":
            return self.focal_loss(masks, preds)
        elif loss_name == "tversky":
            return self.tversky_loss(masks, preds)
        elif loss_name == "combo":
            dice = self.dice_loss(masks, preds)
            ce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(masks, preds))
            return 0.5 * dice + 0.5 * ce
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(weight * bce)

    @staticmethod
    def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        TP = tf.reduce_sum(y_true * y_pred)
        FP = tf.reduce_sum((1 - y_true) * y_pred)
        FN = tf.reduce_sum(y_true * (1 - y_pred))
        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        return 1 - tversky

    # ---------- METRICS ----------
    def compute_metrics(self, preds, masks):
        metrics = {}
        metrics["dice"] = self.dice_coefficient(masks, preds)
        metrics["iou"] = self.iou_score(masks, preds)
        metrics["pixel_acc"] = self.pixel_accuracy(masks, preds)
        metrics["sensitivity"] = self.sensitivity(masks, preds)
        metrics["specificity"] = self.specificity(masks, preds)
        return metrics

    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    def iou_score(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def pixel_accuracy(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        total = tf.size(y_true, out_type=tf.float32)
        return correct / total

    @staticmethod
    def sensitivity(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        TP = tf.reduce_sum(y_true * y_pred)
        FN = tf.reduce_sum(y_true * (1 - y_pred))
        return (TP + smooth) / (TP + FN + smooth)

    @staticmethod
    def specificity(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        TN = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        FP = tf.reduce_sum((1 - y_true) * y_pred)
        return (TN + smooth) / (TN + FP + smooth)
