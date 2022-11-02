from self_attn import SelfAttn
import tensorflow as tf

class Attn(tf.keras.layers.Layer):
    """
    Desc
    """
    def __init__(self):
        pass


if __name__ == "__main__":
    attn = SelfAttn(5, 2);
    X = tf.random.normal((10, 5));

    print(attn.trainable_weights)

    print(attn(X));
