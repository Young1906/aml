from argparse import ArgumentParser
from train import train

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

args = parser.parse_args();

if __name__ == "__main__":
    train(
            path                = args.PATH,
            batch_size          = args.BATCH_SIZE,  
            input_shape         = args.SHAPE,
            backbone_name       = args.BACKBONE,
            embedding_size      = args.EMBEDDING_SIZE,
            margin              = args.MARGIN)

