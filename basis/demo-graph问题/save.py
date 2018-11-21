import tensorflow as tf
from net import net1, net2

def main():
    graph1 = tf.Graph()
    with graph1.as_default():
        x1 = tf.placeholder(shape=(1, 100, 100, 3), dtype=tf.float32)
        y1 = net1(x1)
        saver1 = tf.train.Saver(name="saver")
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver1.save(sess, './net1')

    graph2 = tf.Graph()
    with graph2.as_default():
        x2 = tf.placeholder(shape=(1, 200, 200, 3), dtype=tf.float32)
        y2 = net2(x2)
        saver2 = tf.train.Saver(name="saver")
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver2.save(sess, './net2')

if __name__ == '__main__':
    main()

