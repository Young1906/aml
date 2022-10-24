from typing import List

from pipeline import pipeline
from net import Backbone, SiameseNet
from dataset import get_dataset

def train(
        path                : str,                  # Path to dataset
        batch_size          : int,                  # Size of mini-batch to generate triplets online
        input_shape         : List[int],            # Shape of standardize input into backbone
        backbone_name       : str,                  # Name of backbone
        embedding_size      : int,                  # Size of embedding after foward to backbone
        margin              : float,
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
    gen = get_dataset(
            path            = path,
            batch_size      = batch_size,
            backbone        = backbone);


    for (a, p, n) in gen:
        break


