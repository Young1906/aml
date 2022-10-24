usage:  [-h] [--PATH PATH] [--SHAPE SHAPE [SHAPE ...]] [--BACKBONE BACKBONE]
        [--EMBEDDING_SIZE EMBEDDING_SIZE]

Train a network to recognize face's id

options:
  -h, --help            show this help message and exit
  --PATH PATH           Path/to/data/set/folder
  --SHAPE SHAPE [SHAPE ...]
                        Shape of the image that got fed to the network (None x W x H x C)
  --BACKBONE BACKBONE   Backbone to calculate image's embedding (inception_v3 / mobilenet_v2 /
                        efficientnet)
  --EMBEDDING_SIZE EMBEDDING_SIZE
                        Embedding's size after forward the image via BACKBONE
