from typing import List

from pipeline import pipeline
from net import Backbone, SiameseNet
from dataset import get_dataset
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def train(
        path                : str,                  # Path to dataset
        batch_size          : int,                  # Size of mini-batch to generate triplets online
        input_shape         : List[int],            # Shape of standardize input into backbone
        backbone_name       : str,                  # Name of backbone
        embedding_size      : int,                  # Size of embedding after foward to backbone
        margin              : float,                # Triplet loss margin
        batch_size_ds       : int,                  # Actual training batch size
    ):
    # Construct backbone and siamese network
    backbone = Backbone(
            input_shape     = input_shape,
            backbone_name   = backbone_name,
            embedding_size  = embedding_size);

    # siamese
    siamese = SiameseNet(
            margin          = margin,
            backbone        = backbone);

    # dataset
    # input_shape with batch
    batched_input_shape = [None, ] + input_shape;

    gen = lambda: get_dataset(
            path            = path,
            batch_size      = batch_size,
            backbone        = backbone,
            input_size      = batched_input_shape);

    ds = tf.data.Dataset.from_generator(
                gen,
                output_signature = (
                    tf.TensorSpec(shape = input_shape, dtype = tf.float32),
                    tf.TensorSpec(shape = input_shape, dtype = tf.float32),
                    tf.TensorSpec(shape = input_shape, dtype = tf.float32)
                    ))
    
    ds = ds.batch(batch_size_ds);


    for (a, p, n) in ds:
        print(a.shape);break


