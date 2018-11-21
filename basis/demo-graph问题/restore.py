import tensorflow as tf
import numpy as np
from net import net1, net2

def main():
    # graph1 = tf.Graph()
    graph2 = tf.Graph()
    # 在graph1中定义网络1
    # with graph1.as_default():
    x1 = tf.placeholder(shape=(1, 100, 100, 3), dtype=tf.float32)
    y1 = net1(x1)
    saver1 = tf.train.Saver(name="saver")
    init1 = tf.global_variables_initializer()

    # 在graph2中定义网络2
    with graph2.as_default():
        x2 = tf.placeholder(shape=(1, 200, 200, 3), dtype=tf.float32)
        y2 = net2(x2)
        saver2 = tf.train.Saver(name="saver") # 由于是在另一个graph中，所以这里name可以是saver，前面一样

    # 定义两个Session
    sess1 = tf.Session()
    sess2 = tf.Session(graph=graph2)

    # 你只能在sess1里执行y1, 在sess2里执行y2
    # sess1.run(tf.global_variables_initializer()) 是会报错的，因为tf.global_variables_initializer()
    # 是定义在默认graph上的，而不是graph1上，但sess1 绑定了graph1。
    # 这句话才是可以执行的，但是要restore的话这句话不需要，这里只是给你看一下。
    # sess1.run(init1)
    # restore net1
    saver1.restore(sess1, './net1')
    res1 = sess1.run(y1, feed_dict={x1: np.ones([1, 100, 100, 3])})

    saver2.restore(sess2, './net2')
    res2 = sess2.run(y2, feed_dict={x2: np.ones([1, 200, 200, 3])})

if __name__ == '__main__':
    main()