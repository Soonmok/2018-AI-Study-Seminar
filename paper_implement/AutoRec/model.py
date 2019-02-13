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
            rate=0.8)
        layer_2 = tf.layers.dense(
            inputs=layer_1, 
            units=1024,
            activation=tf.nn.sigmoid,
            name="encode1")
        features = tf.layers.dense(
            inputs=layer_1,
            units=hidden_size,
            activation=tf.nn.sigmoid,
            name="encode2")
        with tf.variable_scope('encode1', reuse=True):
            self.w_encoder_1 = tf.get_variable('kernel')
        with tf.variable_scope('encode2', reuse=True):
            self.w_encoder_2 = tf.get_variable('kernel')
        return features

    # decode encoded feature into predicted rating matrix
    def decode(self, features, recontruction_size):
        layer_1 = tf.layers.dense(
            inputs=features,
            units=1024, 
            name="decode1")
        X_dense_reconstructed = tf.layers.dense(
            inputs=layer_1,
            units=recontruction_size,
            name="decode2")

        with tf.variable_scope('decode1', reuse=True):
                self.w_decoder_1 = tf.get_variable('kernel')
        with tf.variable_scope('decode2', reuse=True):
                self.w_decoder_2 = tf.get_variable('kernel')
        return X_dense_reconstructed
