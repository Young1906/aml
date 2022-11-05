import tensorflow as tf
import numpy as np
from PIL import Image
from typing import List
import cfg
from matplotlib import pyplot as plt
import cv2

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

def detect_face(img):
    faceCascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_alt2.xml");
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.CV_HAAR_SCALE_IMAGE
            )

    _x, _y, _W, _H = 0, 0, 0, 0;

    # Return the largest face
    for (x, y, w, h) in faces:
        if w * h > _W * _H:
            _x, _y, _W, _H = x, y, w, h;

    return _x, _y, _W, _H




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
    img = Image.open(path)    
    
    # face location
    x, y, w, h = detect_face(np.array(img));
    
    flag = 1 if w > 0 else 0;

    if flag == 0:
        return None, flag

    # Convert image to array
    X = np.array(img);

    # Cropping
    X = X[x:x+w, y:y+h, :]

    # Pixel normalization
    X = X / 255. # Pixel normaliztion

    # Apply augmentation
    if augment:
        X = f_augment(X, **kwargs);

    # Resize X to network's input's shape
    X = resize(X, input_size)



    return X, flag


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
    plt.savefig("test.png");
    # plt.show();

