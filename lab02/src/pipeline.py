import tensorflow as tf
import numpy as np
from PIL import Image
from typing import List
import cfg
from matplotlib import pyplot as plt

def resize(
        X                       : tf.Tensor,
        output_size             : List[int])->tf.Tensor:
    """
    output_shape : intented output's shape
        shape : (Batch x Width x Height x Channel)
    """
    _, W, H, _ = output_size 
    layer = tf.keras.layers.Resizing(H, W);

    return layer(X)

def augment(
        X                       : tf.Tensor,
        random_constrast_factor : float):

    augment_layer = tf.keras.layers.RandomContrast(random_contrast_factor, seed = 2022)
    # X = tf.image.random_contrast(X, .5, 1.5)
    return augment_layer(X)


def pipeline(
        path                    : str,
        input_size              : List[int],
        random_contrast_factor  : float)->tf.Tensor:
    """
    Description
    
    Arg
    """
    X = np.array(Image.open(path));
    X = X / 255. # Pixel normaliztion

    # Resize X to network's input's shape
    X = resize(X, input_size)

    # Apply augmentation
    X = augment(X, random_contrast_factor);

    return X


if __name__ == "__main__":
    sample_path = "datasets/lfw_selection/Harrison_Ford/Harrison_Ford_0001.jpg"
    
    input_shape = cfg.INPUT_SHAPE
    random_contrast_factor = cfg.RANDOM_CONTRAST_FACTOR

    X = pipeline(sample_path, input_shape, random_contrast_factor);
    plt.imshow(X);
    plt.show();
