import tensorflow as tf
from tensorflow.contrib.layers import flatten


def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1)
    return tf.Variable(w)


def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)


def LeNet(x):
    # name:      conv5-6    
    # structure: Input = 32x32x1. Output = 28x28x6.
    # weights:   (5*5*1+1)*6
    # connections: (28*28*5*5+28*28)*6
    conv1_W = init_weight((5,5,1,6))
    conv1_b = init_bias(6)
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    #Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    #conv5-16
    #input 14x14x6 Output = 10x10x16.
    #weights: (5*5*6+1)*16 ---real Lenet-5 is (5*5*3+1)*6+(5*5*4+1)*9+(5*5*6+1)*1
    conv2_W = init_weight((5, 5, 6, 16))
    conv2_b = init_bias(16)
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    #Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #Input = 5x5x16. Output = 400.

    conv3_W = init_weight((5, 5, 16, 64))
    conv3_b = init_bias(64)
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 5, 5, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    fc0   = flatten(conv3)


    #Input = 84. Output = 10.
    fc3_W  = init_weight((64,10))
    fc3_b  = init_bias(10)
    logits = tf.matmul(fc0, fc3_W) + fc3_b
    
    return logits, fc0
