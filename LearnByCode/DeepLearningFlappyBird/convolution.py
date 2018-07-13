# 학습을 위한 툴인 텐서플로우를 불러옵니다
import tensorflow as tf

ACTIONS = 2

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def create_cnn_layer(data, weight, bias, stride):
    weight = weight_variable(weight)
    bias = bias_variable(bias)

    return tf.nn.relu(conv2d(data, weight, stride) + bias)

def create_fc_layer(data, weight, bias):
    weight = weight_variable(weight)
    bias = bias_variable(bias)

    return tf.nn.relu(tf.matmul(data, weight) + bias)


def createNetwork():
    
    data = tf.placeholder("float", [None, 80, 80, 4])

    h_conv1 = create_cnn_layer(data, weight=[8, 8, 4, 32], bias=[32], stride=4)

    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = create_cnn_layer(h_pool1, weight=[4, 4, 32, 64], bias=[64], stride=2)

    h_conv3 = create_cnn_layer(h_conv2, weight=[3, 3, 64, 64], bias=[64], stride=1)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = create_fc_layer(h_conv3_flat, weight=[1600, 512], bias=[512])

    readout = create_fc_layer(h_fc1, weight=[512, ACTIONS], bias=[ACTIONS])

    return data, readout, h_fc1, h_conv1, h_pool1, h_conv2, h_conv3, h_conv3_flat
