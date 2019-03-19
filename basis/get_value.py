import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("1.jpg")
img = cv2.resize(img, (224, 224))
input = np.array([img])
a = tf.placeholder(tf.float32, (1, 224, 224, 3), name='value')

def net(x):
    y = tf.layers.conv2d(a, 64, 3, name='conv1')
    return tf.layers.conv2d(y, 3, 3, name='conv2')

b = net(a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b, feed_dict={a:input})
    writer = tf.summary.FileWriter('./logs', sess.graph)
    writer.close()
    c = tf.get_default_graph().get_tensor_by_name(name='conv1/BiasAdd:0')
    # c = tf.get_default_graph.get_operation_by_name(name='conv1/BiasAdd')
    print(sess.run(c, feed_dict={a:input}))
    d = 1
