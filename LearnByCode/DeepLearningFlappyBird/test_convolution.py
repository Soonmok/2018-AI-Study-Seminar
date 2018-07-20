import convolution as cnn
import numpy as np
import tensorflow as tf

data = tf.placeholder("float", [32, 80, 80, 4])
weight_shape = [8, 8, 4, 32]
bias_shape = [32]

def test_weight_variable():
    weight = cnn.weight_variable(weight_shape)
    assert weight.shape == [8, 8, 4, 32], \
           "return 값의 shape이 parameter 값과 같아야 합니다"

def test_bias_variable():
    bias = cnn.bias_variable(bias_shape)
    assert bias.shape == [32],\
           "return값의 shape이 parameter값과 같아야 합니다"

def test_conv2d():
    conv2d = cnn.conv2d(data, weight=weight_shape, stride=4, bias=bias_shape)
    assert conv2d.shape == (32, 20, 20, 32),\
    "output shape -> (32, n, n, 32), \
    n = ((width - filter_width + (2 * padding)) / stride) + 1, \
    padding = (filter_width - stride) / 2"

def test_create_cnn_layer():
    h_conv1 = cnn.create_cnn_layer(data, weight=[8, 8, 4, 32], bias=[32], stride=4)    
    assert h_conv1.shape == (32, 20, 20, 32), \
    "conv2d 함수가 통과되었다면, output shape는 conv2d와 같아야 합니다"

def test_create_fc_layer():
    data2 = tf.placeholder("float", [32, 1600])
    h_fc1 = cnn.create_fc_layer(data2, weight=[1600, 512], bias=[512])
    assert h_fc1.shape == (32, 512), \
    "1600개의 노드를 512개의 노드로 바꾸어야 합니다 -> output shape->(32, 512)"

def test_create_network():
    _, readout, h_fc1, h_conv1, h_pool1, h_conv2, h_conv3, h_conv3_flat = cnn.createNetwork() 
    assert(h_conv1.shape == (32, 20, 20, 32))
    assert(h_pool1.shape == (32, 10, 10, 32))
    assert(h_conv2.shape == (32, 5, 5, 64))
    assert(h_conv3.shape == (32, 5, 5, 64))
    assert(h_conv3_flat.shape == (32, 1600))
    assert(h_fc1.shape == (32, 512))
    assert(readout.shape == (32, 2))

if __name__ == "__main__":
    test_weight_variable()
    test_bias_variable()
    test_create_cnn_layer()
    test_create_network()
