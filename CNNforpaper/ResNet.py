import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np

BN_DECAY = 0.9

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_data():
    dict_train = unpickle('cifar-10-python\data_batch_1')
    train_data = dict_train[b'data']
    train_labels = dict_train[b'labels']
    dict_train = unpickle('cifar-10-python\data_batch_2')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('cifar-10-python\data_batch_3')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('cifar-10-python\data_batch_4')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('cifar-10-python\data_batch_5')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_test = unpickle('cifar-10-python\\test_batch')
    test_data = dict_test[b'data']
    test_labels = dict_test[b'labels']
    train_data = np.reshape(train_data, [-1, 32, 32, 3], 'F')
    train_data = np.transpose(train_data, [0, 2, 1, 3])
    test_data = np.reshape(test_data, [-1, 32, 32, 3], 'F')
    test_data = np.transpose(test_data, [0, 2, 1, 3])
    return train_data, train_labels, test_data, test_labels


def weight_variable(shape):
  initial = tf.random_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(x, shape):
    initial = tf.constant(x, shape=shape)
    return tf.Variable(initial)

def batch_norm(x, is_training):
    global BN_DECAY
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
        # tf.Variable(tf.constant(tf.zeros_initializer(), shape=params_shape), 'beta')
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection('store_mean', update_moving_mean)
    tf.add_to_collection('store_variance', update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 'Batch_norm')

def conv(x, Weight_shape, bias_val, bias_shape, stride):
    W_conv = weight_variable(Weight_shape)
    b_conv = bias_variable(bias_val, bias_shape)
    out_conv = tf.nn.conv2d(x, W_conv, strides=stride, padding="SAME") + b_conv
    out_BN = batch_norm(out_conv, is_training=True)
    return out_BN

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def deep(x, is_training):
    # with tf.name_scope("reshape"):
    #     x_orig = tf.reshape(x, [-1, 32, 32, 3])

    with tf.name_scope("pre_conv"):
        with tf.name_scope("conv"):
            W_preconv = weight_variable([3, 3, 3, 16])
            b_preconv = bias_variable(0., [16])
            preconv = tf.nn.conv2d(x, W_preconv, strides=[1, 1, 1, 1],
                                   padding="SAME") + b_preconv
            preBN = batch_norm(preconv, is_training)

        out_preconv = tf.nn.relu(preBN)

    with tf.name_scope("Block1_16"):
        with tf.name_scope("conv1"):
            W_conv1 = weight_variable([3, 3, 16, 16])
            b_conv1 = bias_variable(0., [16])
            conv1 = tf.nn.conv2d(out_preconv, W_conv1, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv1
            BN1 = batch_norm(conv1, is_training)

        out_conv1 = tf.nn.relu(BN1)

        with tf.name_scope("conv2"):
            W_conv2 = weight_variable([3, 3, 16, 16])
            b_conv2 = bias_variable(0., [16])
            conv2 = tf.nn.conv2d(out_conv1, W_conv2, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv2
            BN2 = batch_norm(conv2, is_training)

        out_conv2 = tf.add(BN2, out_preconv)
        out_block1_16 = tf.nn.relu(out_conv2)

    with tf.name_scope("Block2_16"):
        with tf.name_scope("conv3"):
            W_conv3 = weight_variable([3, 3, 16, 16])
            b_conv3 = bias_variable(0., [16])
            conv3 = tf.nn.conv2d(out_block1_16, W_conv3, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv3
            BN3 = batch_norm(conv3, is_training)

        out_conv3 = tf.nn.relu(BN3)

        with tf.name_scope("conv4"):
            W_conv4 = weight_variable([3, 3, 16, 16])
            b_conv4 = bias_variable(0., [16])
            conv4 = tf.nn.conv2d(out_conv3, W_conv4, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv4
            BN4 = batch_norm(conv4, is_training)

        out_conv4 = tf.add(BN4, out_block1_16)
        out_block2_16 = tf.nn.relu(out_conv4)

    with tf.name_scope("Block3_16"):
        with tf.name_scope("conv5"):
            W_conv5 = weight_variable([3, 3, 16, 16])
            b_conv5 = bias_variable(0., [16])
            conv5 = tf.nn.conv2d(out_block2_16, W_conv5, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv5
            BN5 = batch_norm(conv5, is_training)

        out_conv5 = tf.nn.relu(BN5)

        with tf.name_scope("conv6"):
            W_conv6 = weight_variable([3, 3, 16, 16])
            b_conv6 = bias_variable(0., [16])
            conv6 = tf.nn.conv2d(out_conv5, W_conv6, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv6
            BN6 = batch_norm(conv6, is_training)

        out_conv6 = tf.add(BN6, out_block2_16)
        out_block3_16 = tf.nn.relu(out_conv6)

    with tf.name_scope("Block1_32"):
        with tf.name_scope("conv32-1"):
            W_conv32_1 = weight_variable([3, 3, 16, 32])
            b_conv32_1 = bias_variable(0., [32])
            conv32_1 = tf.nn.conv2d(out_block3_16, W_conv32_1, strides=[1, 2, 2, 1],
                                 padding="SAME") + b_conv32_1
            BN32_1 = batch_norm(conv32_1, is_training)

        out_conv32_1 = tf.nn.relu(BN32_1)

        with tf.name_scope("conv32-2"):
            W_conv32_2 = weight_variable([3, 3, 32, 32])
            b_conv32_2 = bias_variable(0., [32])
            conv32_2 = tf.nn.conv2d(out_conv32_1, W_conv32_2, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_2
            BN32_2 = batch_norm(conv32_2, is_training)

        with tf.name_scope("shortcut1"):
            W_SC1 = weight_variable([1, 1, 16, 32])
            b_SC1 = bias_variable(0., [32])
            conv_SC1 = tf.nn.conv2d(out_block3_16, W_SC1, strides=[1, 2, 2, 1],
                                    padding="SAME") + b_SC1
            out_SC1 = batch_norm(conv_SC1, is_training)

        out_conv32_2 = tf.add(BN32_2, out_SC1)
        out_block1_32 = tf.nn.relu(out_conv32_2)

    with tf.name_scope("Block2_32"):
        with tf.name_scope("conv32-3"):
            W_conv32_3 = weight_variable([3, 3, 32, 32])
            b_conv32_3 = bias_variable(0., [32])
            conv32_3 = tf.nn.conv2d(out_block1_32, W_conv32_3, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_3
            BN32_3 = batch_norm(conv32_3, is_training)

        out_conv32_3 = tf.nn.relu(BN32_3)

        with tf.name_scope("conv32-4"):
            W_conv32_4 = weight_variable([3, 3, 32, 32])
            b_conv32_4 = bias_variable(0., [32])
            conv32_4 = tf.nn.conv2d(out_conv32_3, W_conv32_4, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_4
            BN32_4 = batch_norm(conv32_4, is_training)

        out_conv32_4 = tf.add(BN32_4, out_block1_32)
        out_block2_32 = tf.nn.relu(out_conv32_4)


    with tf.name_scope("Block3_32"):
        with tf.name_scope("conv32-5"):
            W_conv32_5 = weight_variable([3, 3, 32, 32])
            b_conv32_5 = bias_variable(0., [32])
            conv32_5 = tf.nn.conv2d(out_block2_32, W_conv32_5, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_5
            BN32_5 = batch_norm(conv32_5, is_training)

        out_conv32_5 = tf.nn.relu(BN32_5)

        with tf.name_scope("conv32-6"):
            W_conv32_6 = weight_variable([3, 3, 32, 32])
            b_conv32_6 = bias_variable(0., [32])
            conv32_6 = tf.nn.conv2d(out_conv32_5, W_conv32_6, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_6
            BN32_6 = batch_norm(conv32_6, is_training)

        out_conv32_6 = tf.add(BN32_6, out_block2_32)
        out_block3_32 = tf.nn.relu(out_conv32_6)

    with tf.name_scope("Block1_64"):
        with tf.name_scope("conv64-1"):
            W_conv64_1 = weight_variable([3, 3, 32, 64])
            b_conv64_1 = bias_variable(0., [64])
            conv64_1 = tf.nn.conv2d(out_block3_32, W_conv64_1, strides=[1, 2, 2, 1],
                                 padding="SAME") + b_conv64_1
            BN64_1 = batch_norm(conv64_1, is_training)

        out_conv64_1 = tf.nn.relu(BN64_1)

        with tf.name_scope("conv64-2"):
            W_conv64_2 = weight_variable([3, 3, 64, 64])
            b_conv64_2 = bias_variable(0., [64])
            conv64_2 = tf.nn.conv2d(out_conv64_1, W_conv64_2, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv64_2
            BN64_2 = batch_norm(conv64_2, is_training)

        with tf.name_scope("shortcut2"):
            W_SC2 = weight_variable([1, 1, 32, 64])
            b_SC2 = bias_variable(0., [64])
            conv_SC2 = tf.nn.conv2d(out_block3_32, W_SC2, strides=[1, 2, 2, 1],
                                    padding="SAME") + b_SC2
            out_SC2 = batch_norm(conv_SC2, is_training)

        out_conv64_2 = tf.add(BN64_2, out_SC2)
        out_block1_64 = tf.nn.relu(out_conv64_2)

    with tf.name_scope("Block2_64"):
        with tf.name_scope("conv64-3"):
            W_conv64_3 = weight_variable([3, 3, 64, 64])
            b_conv64_3 = bias_variable(0., [64])
            conv64_3 = tf.nn.conv2d(out_block1_64, W_conv64_3, strides=[1, 1, 1, 1],
                                    padding="SAME") + b_conv64_3
            BN64_3 = batch_norm(conv64_3, is_training)

        out_conv64_3 = tf.nn.relu(BN64_3)

        with tf.name_scope("conv64-4"):
            W_conv64_4 = weight_variable([3, 3, 64, 64])
            b_conv64_4 = bias_variable(0., [64])
            conv64_4 = tf.nn.conv2d(out_conv64_3, W_conv64_4, strides=[1, 1, 1, 1],
                                    padding="SAME") + b_conv64_4
            BN64_4 = batch_norm(conv64_4, is_training)

        out_conv64_4 = tf.add(BN64_4, out_block1_64)
        out_block2_64 = tf.nn.relu(out_conv64_4)

    with tf.name_scope("Block3_64"):
        with tf.name_scope("conv64-5"):
            W_conv64_5 = weight_variable([3, 3, 64, 64])
            b_conv64_5 = bias_variable(0., [64])
            conv64_5 = tf.nn.conv2d(out_block2_64, W_conv64_5, strides=[1, 1, 1, 1],
                                    padding="SAME") + b_conv64_5
            BN64_5 = batch_norm(conv64_5, is_training)

        out_conv64_5 = tf.nn.relu(BN64_5)

        with tf.name_scope("conv64-6"):
            W_conv64_6 = weight_variable([3, 3, 64, 64])
            b_conv64_6 = bias_variable(0., [64])
            conv64_6 = tf.nn.conv2d(out_conv64_5, W_conv64_6, strides=[1, 1, 1, 1],
                                    padding="SAME") + b_conv64_6
            BN64_6 = batch_norm(conv64_6, is_training)

        out_conv64_6 = tf.add(BN64_6, out_block2_64)
        out_block3_64 = tf.nn.relu(out_conv64_6)

    with tf.name_scope('pool'):
        pooling = max_pool_2x2(out_block3_64)

    with tf.name_scope("fc"):
        W_fc = weight_variable([4 * 4 * 64, 10])
        b_fc = bias_variable(0., [10])
        reshape = tf.reshape(pooling, [-1, 4 * 4 * 64])
        y_conv = tf.matmul(reshape, W_fc) + b_fc

    return y_conv

def main():
    train_data, train_labels, test_data, test_labels = read_data()
    is_train = True

    x = tf.placeholder(tf.float32, [None, 32*32*3], name='input_image')
    y = tf.placeholder(tf.int32, [None], name='label')

    y_pred = deep(x, is_train)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred))

    with tf.name_scope('optimize'):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.piecewise_constant(global_step, [1000, 2000], [0.01, 0.001, 0.0001])
        train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs", sess.graph)
        writer.flush()
        writer.close()
        sess.run(tf.global_variables_initializer())

        # for i in range(20000):
        #     batch = mnist.train.next_batch(100)
        #     if i % 100 == 0:
        #         train_accuracy = accuracy.eval(feed_dict={
        #           x: batch[0], y_: batch[1], keep_prob: 1.0})
        #         print('step %d, training accuracy %g' % (i, train_accuracy))
        #         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #
        # print('test accuracy %g' % accuracy.eval(feed_dict={
        #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    main()