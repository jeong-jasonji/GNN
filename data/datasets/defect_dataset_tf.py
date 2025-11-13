import tensorflow as tf
import os
from core.base_dataset import BaseDataset
from core.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("defect_images")
class DefectDataset(BaseDataset):
    """Simple image folder dataset structure:
       data_dir/
         train/
           good/
           defective/
         val/
           good/
           defective/
    """

    def prepare_data(self):
        # Can add preprocessing or verification here
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"{self.data_dir} not found")

    def preprocess(self, img, label):
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def load_dataset(self, split="train"):
        ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(self.data_dir, split),
            batch_size=self.batch_size,
            image_size=(224, 224),
            shuffle=True
        )
        if self.augment and split == "train":
            ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
