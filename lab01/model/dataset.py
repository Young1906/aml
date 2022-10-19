import tensorflow_datasets as tfds


# Dataset 
def getDataset(f):
    """
    f: Preprocessing Function
    """
    (train, valid, test), info = tfds.load(
        name = "tf_flowers", 
        split = ["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised = True,
        with_info = True
    )

    # Number of class in the dataset
    num_classes = info.features["label"].num_classes

    # Mapp preprocessing function onto the dataset
    train = train.map(f)
    valid = valid.map(f)
    test = test.map(f)

    return (train, valid, test), num_classes

