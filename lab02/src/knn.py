import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from net import Backbone
from dataset import parse_y
from utils import CateEncoder
from pipeline import pipeline
from typing import List
import glob, random, os
from argparse import ArgumentParser 
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

# Training params:
parser = ArgumentParser(
        description = "Using KNN on Embedding of each image calculated by Backbone that is trained by Siamese Net");

parser.add_argument(
        "--PATH",
        type = str,
        help = "Path/to/image/folder"
)
parser.add_argument(
        "--SHAPE",
        nargs = "+",
        type = int,
        help = "Shape of the image that got fed to the network (None x W x H x C)"
)

parser.add_argument(
        "--BACKBONE",
        type = str,
        help = "Backbone to calculate image's embedding (inception_v3 / mobilenet_v2 / efficientnet)")

parser.add_argument(
        "--EMBEDDING_SIZE",
        type = int,
        help = "Embedding's size after forward the image via BACKBONE")


parser.add_argument(
        "--CHECKPOINT",
        type = str,
        help = "Backbone checkpoint")

args = parser.parse_args();


def get_model(
        input_shape         : List[int],
        backbone_name       : str,
        embedding_size      : int,
        checkpoint_pth      : str) -> tf.keras.Model:
    
    backbone = Backbone(
            input_shape         = input_shape,
            backbone_name       = backbone_name,
            embedding_size      = embedding_size) 

    # Load weight
    backbone.load_weights(checkpoint_pth);

    return backbone;


def knn(
        # path to image folder
        pth                 : str,
        
        # Backbone params
        input_shape         : List[int],
        backbone_name       : str,
        embedding_size      : int,
        checkpoint_pth      : str,

        # KNN params


        ):
    # List of all files
    ls_files = glob.glob(f"{pth}/*/*");
    
    # number of sample
    N = len(ls_files);

    # shuffle
    random.shuffle(ls_files);

    # list of all classes
    _classes = os.listdir(pth);

    # cate encoder
    ce = CateEncoder(_classes);

    # Label
    y = [parse_y(x, ce) for x in ls_files];
    y = np.array(y);

    # Embedding of all images;
    E = np.zeros((N, embedding_size));

    with tf.device("/gpu:0"):
        # Loading backbone
        backbone = get_model(
                input_shape     = input_shape,
                backbone_name   = backbone_name,
                embedding_size  = embedding_size,
                checkpoint_pth  = checkpoint_pth)

    # Calculate embedding
    for i, fn in tqdm(enumerate(ls_files), desc = "Calculating embedding"):
        X = pipeline(
                path        = fn,
                input_size  = [None,] + input_shape,
                augment     = False)

        with tf.device("/gpu:0"):
            embedding = backbone(X);

        E[i,:] = embedding.numpy();

    # Train KNN
    X_train, X_test, y_train, y_test = train_test_split(E, y, test_size = .3);

    # KNN classifier
    knn = KNeighborsClassifier();
    knn.fit(X_train, y_train);

    y_pred = knn.predict(X_test);
    print(f"Test set acc : {(y_pred==y_test).mean():.4f}");

    # confusion matrix
    cf = confusion_matrix(y_test, y_pred);

    sns.heatmap(cf);
    plt.savefig("heatmap.png");

    





if __name__ == "__main__":knn(
        pth                 = args.PATH,
        input_shape         = args.SHAPE,
        backbone_name       = args.BACKBONE,
        embedding_size      = args.EMBEDDING_SIZE,
        checkpoint_pth      = args.CHECKPOINT,
        )

