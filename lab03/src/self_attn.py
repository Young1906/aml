import tensorflow as tf
from typing import List

def xavier_init(shape:List[int])->tf.Tensor:
    """
    """
    assert len(shape) == 2, NotImplementedError("len(shape) != 2");
    n, _ = shape;

    lower, upper = -1 / n**.5, 1 / n**.5
    return tf.random.uniform(shape, minval = lower, maxval = upper);

class SelfAttn(tf.keras.layers.Layer):
    """
    Desc:
        Implementation of self attention layer
    """
    def __init__(self, input_size, output_size):
        super().__init__();

        # K, V, Q
        self.K = tf.Variable(initial_value =  xavier_init((input_size, output_size)),
                trainable = True,
                name = "self_attn::key")
        self.V = tf.Variable(initial_value =  xavier_init((input_size, output_size)),
                trainable = True,
                name = "self_attn::value")
        self.Q = tf.Variable(initial_value =  xavier_init((input_size, output_size)),
                trainable = True,
                name = "self_attn::query")

    def call(self, X):
        """
        Xs : sequence of input, shape = (seq_len x input_size)
        """
        # Calculate seq of K, V, and Q
        K = X @ self.K # (seq_len x output_size)
        V = X @ self.V
        Q = X @ self.Q

        # Scale dot product
        attn = tf.nn.softmax(K @ tf.transpose(Q), axis = -1) @ V
        return attn
