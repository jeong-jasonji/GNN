import tensorflow as tf
from core.builder import load_config, build_model, build_dataset

cfg = load_config("configs/train_unet.yaml")
model = build_model(cfg)
dataset = build_dataset(cfg)["val"]

model.load_weights("experiments/runs/defect_segmentation_unet/best_model.h5")

dice_scores, ious = [], []
for x, y in dataset:
    preds = model(x, training=False)
    metrics = model.compute_metrics(preds, y)
    dice_scores.append(metrics["dice"])
    ious.append(metrics["iou"])

print(f"Mean Dice: {tf.reduce_mean(dice_scores):.4f}, Mean IoU: {tf.reduce_mean(ious):.4f}")
