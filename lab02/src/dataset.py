import tensorflow as tf
from typing import List

print(tf.__version__);
print(tf.config.list_physical_devices("GPU"))


def triplet_selection(
        X: List[str],   # List of path to image
        y: List[int],   # Corresponding label
        backbone: tf.keras.Model
    ):


    







