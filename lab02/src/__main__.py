from argparse import ArgumentParser 
from train import train
from matplotlib import pyplot as plt
import pickle

# Training params:
parser = ArgumentParser(
        description = "Train a network to recognize face's id");

parser.add_argument(
        "--PATH",
        type = str,
        help = "Path/to/data/set/folder")

parser.add_argument(
        "--BATCH_SIZE", 
        type = int,
        help = "Mini-batch's size to generate triplets")

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
        "--MARGIN",
        type = float,
        help = "Margin for Triplet Loss Function")

parser.add_argument(
        "--BATCH_SIZE_DS",
        type = int,
        help = "Actual batch' size during training")

parser.add_argument(
        "--EPOCH",
        type = int,
        help = "Number of epochs")

parser.add_argument(
        "--N_VALID",
        type = int,
        help = "Number of hold-out for validation")

parser.add_argument(
        "--LEARNING_RATE",
        type = float,
        help = "Learning rate")

args = parser.parse_args();

if __name__ == "__main__":
    hist, backbone = train(
            path                = args.PATH,
            batch_size          = args.BATCH_SIZE,  
            input_shape         = args.SHAPE,
            backbone_name       = args.BACKBONE,
            embedding_size      = args.EMBEDDING_SIZE,
            margin              = args.MARGIN,
            batch_size_ds       = args.BATCH_SIZE_DS,
            epoch               = args.EPOCH,
            n_valid             = args.N_VALID,
            learning_rate       = args.LEARNING_RATE)

    plt.plot(hist["loss"]["train"]);
    plt.title("Train Loss by Batch");
    plt.savefig("train_loss.png");

    plt.clf();

    plt.plot(hist["loss"]["valid"]);
    plt.title("Validation Loss by Epoch")
    plt.savefig("valid_loss.png");

    # Save loss object
    with open("hist.pkl", "wb") as f:
        pickle.dump(hist, f);

    backbone.save_weights(f"checkpoints/{args.BACKBONE}");


    

