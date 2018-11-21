import tensorflow.contrib.layers as layers

def net1(x):
    y = layers.conv2d(x, 10, 3)
    return y

def net2(x):
    y = layers.conv2d(x, 20, 5)
    return y
