import tensorflow as tf 
import numpy as np 


class AutoEncoder(object):
    def __init__(self, X):
        features = self.encode(X)
        X_reconstructed = self.decode(features)
        self.X_reconstructed = X_reconstructed
        self.features = features
    
    def encode(self, x_input):
        Encode_layer_1 = tf.layers.dense(inputs=x_input, units=10000, activation=tf.nn.relu)
        Encode_layer_2 = tf.layers.dense(inputs=Encode_layer_1, units=1024, activation=tf.nn.relu)
        features = tf.layers.dense(inputs=layer_2, units=512)
        return features

    def decode(self, features):
        Decode_layer_1 = tf.layers.dense(inputs=features, units=1024, activation=tf.nn.relu)
        X_reconstructed = tf.layers.dense(inputs=Decode_layer_1, units=10000, activation=tf.nn.relu)
        return X_reconstructed