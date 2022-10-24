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

def f_augment(
        X                       : tf.Tensor,
        **kwargs): 

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(factor = kwargs.get("random_contrast_factor")),
        tf.keras.layers.RandomFlip(mode = kwargs.get("flip_mode")),
    ])
    return augment(X)


def pipeline(
        path                    : str,
        input_size              : List[int],
        augment                 : bool      = False,
        **kwargs
    )->tf.Tensor:
    """
    Description:
        + read file into numpy array from path
        + pixel normalization
        + augmentation
    
    Arg
        + path                  <str>               : path to image
        + input_size            <list<int>>         : image size to fit into the network
        + augment               <bool>              : whether to augment the image
        + **kwargs                                  : augmentation's parameters
    """
    X = np.array(Image.open(path));

    # Pixel normalization
    X = X / 255. # Pixel normaliztion

    # Convert to gray_scale image
    # X = tf.image.rgb_to_grayscale(X)

    # Apply augmentation
    if augment:
        X = f_augment(X, **kwargs);

    # Resize X to network's input's shape
    X = resize(X, input_size)

    # X = np.expand_dims(X, 0);

    return X


if __name__ == "__main__":
    sample_path = "datasets/lfw_selection/Harrison_Ford/Harrison_Ford_0001.jpg"
    
    input_shape = cfg.INPUT_SHAPE
    random_contrast_factor = cfg.RANDOM_CONTRAST_FACTOR

    X = pipeline(
        path = sample_path,
        input_size = input_shape,
        augment = True,
        random_contrast_factor = 0.5,
        flip_mode = "horizontal_and_vertical");
    plt.imshow(X);
    plt.show();
