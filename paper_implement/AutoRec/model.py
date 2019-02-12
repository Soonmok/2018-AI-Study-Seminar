import tensorflow as tf 
import numpy as np 


class AutoEncoder(object):
    def __init__(self, X, hidden_size):
        X_dense = tf.sparse.to_dense(X)
        features = self.encode(X_dense, hidden_size)
        X_dense_reconstructed = self.decode(features, X_dense.shape[1])
        self.rating_recontructed = X_dense_reconstructed
        self.features = features
 
    def encode(self, x_input, hidden_size):
        layer_1 = tf.layers.dropout(
            x_input,
            rate=0.1)
        features = tf.layers.dense(
            inputs=layer_1, 
            units=x_input.shape[1],
            activation=tf.nn.sigmoid,
            name="encode")
        with tf.variable_scope('encode', reuse=True):
                self.w_encoder = tf.get_variable('kernel')
        return features

    def decode(self, features, recontruction_size):
        X_dense_reconstructed = tf.layers.dense(
            inputs=features,
            units=recontruction_size, 
            activation=tf.nn.selu,
            name="decode")
        with tf.variable_scope('decode', reuse=True):
                self.w_decoder = tf.get_variable('kernel')
        return X_dense_reconstructed
