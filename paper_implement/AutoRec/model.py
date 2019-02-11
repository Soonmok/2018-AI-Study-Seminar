import tensorflow as tf 
import numpy as np 


class AutoEncoder(object):
    def __init__(self, X, hidden_size):
        features = self.encode(X, hidden_size)
        X_reconstructed = self.decode(features)
        self.rating_recontructed = X_reconstructed
        self.features = features
 
    def encode(self, x_input, hidden_size):
        features = tf.layers.dense(
            inputs=x_input, 
            units=x_input.shape[1],
            activation=tf.nn.sigmoid,
            name="encode")
        self.w_encoder = tf.get_variable("encode/W:0")[0]
        return features

    def decode(self, features, recontruction_size):
        X_reconstructed = tf.layers.dense(
            inputs=features,
            units=recontruction_size, 
            activation=tf.nn.selu,
            name="decode")
        self.w_decoder = tf.get_variable("decode/W:0")[0]
        return X_reconstructed
