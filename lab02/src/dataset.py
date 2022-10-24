import tensorflow as tf
from typing import List
from pipeline import pipeline 
import os, glob
from utils import CateEncoder 

def triplet_generator(
        batch_X: List[str],   # List of path to image
        batch_y: List[int],   # Corresponding label
        backbone: tf.keras.Model,
        **kwargs
    ):
    """
    Description
        Construct a set of triplets from batch of images B, that satisfied 
            p = argmax_{i in B | class(i) == class(a)} dist(a, i)
            n = argmin_{i in B | class(i) != class(a)} dist(a, i)
    Args
        batch_X         <list<path/to/image>>
        batch_y         <list<int>>
        backbone        tf.keras.Model

    Return
        Set of triplets {(a, p, n)_i}_{i=1...k}
    """
    # batch_size 
    N = len(batch_X);

    # Read image batch to memory
    batch_X = [
            pipeline(
                path        = x,
                input_size  = kwargs.get("input_shape"),
                augment = False)

            for x in batch_X
        ];

    batch_y = np.array(batch_y);

    # Compute embedding : numpy array to store embedding
    E = np.zeros((N, backbone.embedding_size));

    for (i, X) in enumerate(batch_X):
        embedding = backbone(X);
        E[i, :] = embedding.numpy();


    for i in range(N):
        a = E[i, :] # Embedding of the anchor;

        # Positive & Negative mask;
        pos_mask = batch_y != batch_y[i];
        neg_mask = batch_y == batch_y[i];

        if pos_mask.sum() == 1: continue; 
        # Skip this image if there is only 1 of its class within this batch

        # Distance vector from anchor a to all other image
        D = np.sum((E - a[np.newaxis, :]) * (E - a[np.newaxis, :]), -1);

        # Positive index
        p = np.argmax(np.ma.masked_array(D, mask = pos_mask));

        # Negative index
        n = np.argmax(np.ma.masked_array(D, mask = neg_mask));

        yield batch_X[i], batch_X[n], batch_X[p];

def get_dataset(
        path            : str,
        batch_size      : List[int],
        backbone        : tf.keras.Model
    ):
    """
    pth: path to dataset 
    """

    # List of all files
    ls_files = glob.glob(f"{path}/*/*");

    # List of all classes;
    _target = os.listdir(path);

    print(_target);
    return

    # Number of batch(s)
    N_BATCHES = len(ls_files) // batch_size + 1;

    for i in range(N_BATCHES):
        batch_files = ls_files[ i * batch_size : (i+1) * batch_size];
        
        # Generate triplet from this mini-batch
        triplets = batch_generator();

        for triplet in triplets:
            yield triplet


def parse_fn(path, ce):
    """
    Description
    """

if __name__ == "__main__":
    get_dataset(
            "datasets/lfw_selection", 
            512)

