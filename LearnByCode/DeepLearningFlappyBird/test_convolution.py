import convolution as cnn
import numpy as np
import tensorflow as tf

data = tf.placeholder("float", [32, 80, 80, 4])

def test_create_cnn_layer():
    h_conv1 = cnn.create_cnn_layer(data, weight=[8, 8, 4, 32], bias=[32], stride=4)    
    assert(h_conv1.shape == (32, 20, 20, 32))


def test_create_network():
    h_conv1 = cnn.create_cnn_layer(data, weight=[8, 8, 4, 32], bias=[32], stride=4)

    h_pool1 = cnn.max_pool_2x2(h_conv1)

    h_conv2 = cnn.create_cnn_layer(h_pool1, weight=[4, 4, 32, 64], bias=[64], stride=2)

    h_conv3 = cnn.create_cnn_layer(h_conv2, weight=[3, 3, 64, 64], bias=[64], stride=1)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = cnn.create_fc_layer(h_conv3_flat, weight=[1600, 512], bias=[512])

    readout = cnn.create_fc_layer(h_fc1, weight=[512, 2], bias=[2])

    assert(h_conv1.shape == (32, 20, 20, 32))
    assert(h_pool1.shape == (32, 10, 10, 32))
    assert(h_conv2.shape == (32, 5, 5, 64))
    assert(h_conv3.shape == (32, 5, 5, 64))
    assert(h_conv3_flat.shape == (32, 1600))
    assert(h_fc1.shape == (32, 512))
    assert(readout.shape == (32, 2))

if __name__ == "__main__":
    test_create_cnn_layer()
    test_create_network()
