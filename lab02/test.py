import tensorflow as tf
import numpy as np
from tqdm import tqdm 

if __name__ == "__main__":
    print(tf.config.list_physical_devices());
    with tf.device("/GPU:0"):
        backbone = tf.keras.applications.inception_v3.InceptionV3(
                include_top = False,
                input_shape=[255, 255, 3])

        backbone.build([255, 255, 3]);
        print(backbone.summary());

        X = np.random.normal(0, 1,(1, 255, 255, 3));

        for _ in tqdm(range(25)): 
            y = backbone(X);

    input("Enter to exit");
