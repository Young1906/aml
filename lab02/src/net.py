import tensorflow as tf;
from typing import List
from matplotlib import pyplot as plt


class DenseBatchNorm(tf.keras.layers.Layer):
    def __init__(
            self, 
            unit    : int,
            rate    : float = .1):
        super().__init__();

        self.dense = tf.keras.layers.Dense(units = unit);
        self.bnorm = tf.keras.layers.BatchNormalization();
        self.drop = tf.keras.layers.Dropout(rate);
    
    def call(self, X):
        out = self.dense(X);
        out = self.bnorm(out);

        return self.drop(out);
        

class Backbone(tf.keras.Model):
    def __init__(
            self,
            input_shape         : List[int],
            backbone_name       : str,
            embedding_size      : int):
        super().__init__()

        if backbone_name == "inception_v3":
            backbone = tf.keras.applications.inception_v3.InceptionV3(
                    include_top = False,
                    input_shape = input_shape
            )
        elif backbone_name == "mobilenet_v2":
            backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
                    include_top = False,
                    input_shape = input_shape
            )
        elif backbone_name == "efficientnet":
            backbone = tf.keras.applications.efficientnet.EfficientNetB2(
                    include_top = False,
                    input_shape = input_shape)
        else:
            raise NotImplementedError(f"{name} is not supported (yet)!!")

        self.net = backbone;
        self.embedding_size = embedding_size;

        self.model = tf.keras.Sequential([
            self.net,
            tf.keras.layers.GlobalAveragePooling2D(),
            DenseBatchNorm(512),
            DenseBatchNorm(256),
            DenseBatchNorm(embedding_size),
        ])

    def call(self, X):
        return self.model(X)


class SiameseNet(tf.keras.Model):
    def __init__(
            self,
            margin          : float,
            backbone        : tf.keras.Model):

        super().__init__();
        self.backbone = backbone;
        self.margin = margin;

    @staticmethod
    def l2(u, v):
        """
        Description
            Euler distance between u & v
            f(u, v) = (u - v)'(u - v)

        Args:
            u, v: either a vector of matrix
        
        Return
            scalar / vector
            
        """
        return tf.math.reduce_sum((u - v) * (u - v), -1);

    def triplet_loss(self, a, p, n):
        """
        Description:


        Args:
            a, p, n : embedding of anchor, positive, and negative sample
        """

        loss_vector = tf.math.maximum(0, self.l2(a, p) - self.l2(a, n) + self.margin);
        return tf.math.reduce_mean(loss_vector)


    def call(self, A, P, N):
        """
        Description
        Args
            A: anchor
            P: positive
            N: negative
        """
        a, p, n = self.backbone(A), self.backbone(P), self.backbone(N);
        return self.triplet_loss(a, p, n);
