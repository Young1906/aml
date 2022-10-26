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

    with tf.device("/device:gpu:0"):
        # Construct backbone and siamese network
        backbone = Backbone(
                input_shape     = input_shape,
                backbone_name   = backbone_name,
                embedding_size  = embedding_size);

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
            "valid" : [],
        },
    }

    if os.environ.get("DEV") == "1":
        return hist, backbone;

    # Main training loop

    pbar = tqdm(range(epoch));
    for e in pbar:
        # Update progress bar
        pbar.set_description(f"Epoch {e+1:2d}");

        # create dataset from generator
        ds = tf.data.Dataset.from_generator(
                    gen,
                    output_signature = (
                        tf.TensorSpec(shape = input_shape, dtype = tf.float32),
                        tf.TensorSpec(shape = input_shape, dtype = tf.float32),
                        tf.TensorSpec(shape = input_shape, dtype = tf.float32)
                        ))
        # Train / Validation Split
        valid = ds.take(n_valid)
        train = ds.skip(n_valid)
        
        # Batching
        train = train.batch(batch_size_ds);
        valid = valid.batch(batch_size_ds);

        b_pbar = tqdm(train) # batch progress bar

        b = 1; #Batch counter
        for (a, p, n) in b_pbar:
            
            # DEV
            if os.environ.get("DEV") == "1" and b >=5: break

            with tf.device("/device:gpu:0"):
                with tf.GradientTape() as tape:
                    loss = siamese(a, p, n);

            b_pbar.set_description(f"\t Batch {b:2d} - Loss {loss.numpy():.3f}");

            with tf.device("/device:gpu:0"):
                # Gradient
                grads = tape.gradient(loss, siamese.trainable_weights);
                opt.apply_gradients(zip(grads, siamese.trainable_weights));
            
            hist["loss"]["train"].append(loss.numpy());

            b+=1;
        val_loss = 0;
        counter = 0;

        # At epoch end, calculate validation loss 
        for (a, p, n) in valid:
            with tf.device("/device:gpu:0"):
                vloss = siamese(a, p, n)

            val_loss += vloss.numpy();
            counter += 1;

        val_loss /= counter;
        hist["loss"]["valid"].append(val_loss);
        pbar.set_description(f"Epoch {e+1:2d} - Loss {loss.numpy():.3f} - Validation loss {val_loss:.3f}");

    return hist, backbone;





        




