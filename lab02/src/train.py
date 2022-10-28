from typing import List

from pipeline import pipeline
from net import Backbone, SiameseNet
from dataset import get_dataset
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

def train(
        path                : str,                  # Path to dataset
        batch_size          : int,                  # Size of mini-batch to generate triplets online
        input_shape         : List[int],            # Shape of standardize input into backbone
        backbone_name       : str,                  # Name of backbone
        embedding_size      : int,                  # Size of embedding after foward to backbone
        margin              : float,                # Triplet loss margin
        batch_size_ds       : int,                  # Actual training batch size
        epoch               : int,                  # Number of epochs to train
        n_valid             : int,                  # Number of hold-out triplet to validate
        learning_rate       : float,                # Optimizer learning rate
    ):

    with tf.device("/gpu:0"):
        # Construct backbone and siamese network
        backbone = Backbone(
                input_shape     = input_shape,
                backbone_name   = backbone_name,
                embedding_size  = embedding_size);

        backbone.build([None, *input_shape]);

        # siamese
        siamese = SiameseNet(
                margin          = margin,
                backbone        = backbone);

        # optimizer
        opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)

        # dataset
        # input_shape with batch
        batched_input_shape = [None, ] + input_shape;

        gen = lambda: get_dataset(
                path            = path,
                batch_size      = batch_size,
                backbone        = backbone,
                input_size      = batched_input_shape);


        # History object
        hist = {
            "loss": {
                "train" : [],
            },
        }

        # create dataset from generator
        ds = tf.data.Dataset.from_generator(
                gen,
                output_signature = (
                    tf.TensorSpec(shape = input_shape, dtype = tf.float32),
                    tf.TensorSpec(shape = input_shape, dtype = tf.float32),
                    tf.TensorSpec(shape = input_shape, dtype = tf.float32)
                   ));
        ds = ds.batch(batch_size_ds);


        # Main training loop
        pbar = tqdm(range(epoch));
        for e in pbar:
            # Update progress bar
            pbar.set_description(f"Epoch {e:2d}");

            b_pbar = tqdm(ds) # batch progress bar

            b = 1; #Batch counter
            for (a, p, n) in b_pbar:
                
                # DEV
                if os.environ.get("DEV") == "1" and b >=5: break

                with tf.GradientTape() as tape:
                    loss = siamese(a, p, n);

                b_pbar.set_description(f"\t Batch {b:2d} - Loss {loss.numpy():.3f}");

                # Gradient
                grads = tape.gradient(loss, siamese.trainable_weights);
                opt.apply_gradients(zip(grads, siamese.trainable_weights));
                
                hist["loss"]["train"].append(loss.numpy());

                b+=1;
            
            counter = 0;
    return hist, backbone;
