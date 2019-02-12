import tensorflow as tf 
import numpy as np 


class AutoEncoder(object):
    """ encode and decode to generate predicted rating matrix
    Args : 
        X_dense : batch of rating matrix
        hidden_size : setting number of encoded feature size
    Returns :
        self.X_dense_reconstructed : predicted rating matrix
        self.w_encoder : encoder variables 
        self.w_decoder : decoder variables"""

    # predict rating matrix
    def __init__(self, X_dense, hidden_size):
        features = self.encode(X_dense, hidden_size)
        self.X_dense_reconstructed = self.decode(features, X_dense.shape[1])
    
    # encode rating matrix into small size feature
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

    # decode encoded feature into predicted rating matrix
    def decode(self, features, recontruction_size):
        X_dense_reconstructed = tf.layers.dense(
            inputs=features,
            units=recontruction_size, 
            name="decode")
        with tf.variable_scope('decode', reuse=True):
                self.w_decoder = tf.get_variable('kernel')
        return X_dense_reconstructed
