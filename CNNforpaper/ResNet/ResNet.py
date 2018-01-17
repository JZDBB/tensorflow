#some problems
#1、average pooling strides and ksize and padding is not refered
#2、weight initial I took std = sqrt(2./(weightsize)) is that right?
#3、train method is still not understood =、= mini batch size is 128（done）
#4、bacth_norm with is_training is not understood yet
#5、weight decay have not been solved. weight decay is 0.0001
#6、how to split train data and validate data? 45k/5k ——tra/val (不分)
#7、data augmentation needed.test use orignal image
#8、change data to TFrecord （done）
#9、variable_scope和name_scope区别
#10、BN_decdy = 0.5 -> 0.999?
#11、去掉bias？

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np
import math
import os
import matplotlib.pyplot as plt

BN_DECAY = 0.9
Weight_decay = 0.0001

def read_and_decode(filename, train):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    if train:
        img = tf.image.pad_to_bounding_box(img, 4, 4, 40, 40)
        img = tf.random_crop(img, [32, 32, 3])
        img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int32)

    return img, label

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    data = np.ndarray(shape=(0, 32 * 32 * 3), dtype=np.float32)
    labels = np.ndarray(shape=0, dtype=np.int64)
    for i in range(5):
        tmp = unpickle(os.path.join("cifar-10-python", "data_batch_{}".format(i + 1)))
        data = np.append(data, tmp[b'data'], axis=0)
        labels = np.append(labels, tmp[b'labels'], axis=0)
        print('load training data: data_batch_{}'.format(i + 1))
    data = np.reshape(data, [-1, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
    return data, labels


def load_valid_data():
    tmp = unpickle(os.path.join("cifar-10-python", "test_batch"))
    data = np.ndarray(shape=(0, 32 * 32 * 3), dtype=np.float32)
    labels = np.ndarray(shape=0, dtype=np.int64)

    data = np.append(data, tmp[b'data'], axis=0)
    # data = tmp[b'data']
    labels = np.append(labels, tmp[b'labels'])

    # data.astype(np.float32)
    # labels.astype(np.int64)

    data = np.reshape(data, [-1, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
    print('load test data: test_batch')
    return data, labels

def weight_variable(shape, std):
  initial = tf.random_normal(shape, stddev=std)
  return tf.Variable(initial)

def bias_variable(x, shape):
    initial = tf.constant(x, shape=shape)
    return tf.Variable(initial)


def batch_norm(x, is_training):
    # return tf.contrib.layers.batch_norm(x, is_training=is_training)
    is_training = tf.convert_to_tensor(is_training)
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
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001, 'Batch_norm')


def deep(x, is_training):
    # with tf.name_scope("reshape"):
    #     x_orig = tf.reshape(x, [-1, 32, 32, 3])

    with tf.variable_scope("pre_conv"):
        with tf.variable_scope("conv"):
            W_preconv = weight_variable([3, 3, 3, 16], math.sqrt(2.0/(3*3*3)))
            tf.add_to_collection('w', W_preconv)
            # b_preconv = bias_variable(0., [16])
            preconv = tf.nn.conv2d(x, W_preconv, strides=[1, 1, 1, 1],
                                   padding="SAME") #+ b_preconv
            preBN = batch_norm(preconv, is_training)

        out_preconv = tf.nn.relu(preBN)

    with tf.variable_scope("Block1_16"):
        with tf.variable_scope("conv1"):
            W_conv1 = weight_variable([3, 3, 16, 16], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv1)
            # b_conv1 = bias_variable(0., [16])
            conv1 = tf.nn.conv2d(out_preconv, W_conv1, strides=[1, 1, 1, 1],
                                 padding="SAME") #+ b_conv1
            BN1 = batch_norm(conv1, is_training)

        out_conv1 = tf.nn.relu(BN1)

        with tf.variable_scope("conv2"):
            W_conv2 = weight_variable([3, 3, 16, 16], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv2)
            # b_conv2 = bias_variable(0., [16])
            conv2 = tf.nn.conv2d(out_conv1, W_conv2, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv2
            BN2 = batch_norm(conv2, is_training)

        out_conv2 = tf.add(BN2, out_preconv)
        out_block1_16 = tf.nn.relu(out_conv2)

    with tf.variable_scope("Block2_16"):
        with tf.variable_scope("conv3"):
            W_conv3 = weight_variable([3, 3, 16, 16], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv3)
            # b_conv3 = bias_variable(0., [16])
            conv3 = tf.nn.conv2d(out_block1_16, W_conv3, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv3
            BN3 = batch_norm(conv3, is_training)

        out_conv3 = tf.nn.relu(BN3)

        with tf.variable_scope("conv4"):
            W_conv4 = weight_variable([3, 3, 16, 16], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv4)
            # b_conv4 = bias_variable(0., [16])
            conv4 = tf.nn.conv2d(out_conv3, W_conv4, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv4
            BN4 = batch_norm(conv4, is_training)

        out_conv4 = tf.add(BN4, out_block1_16)
        out_block2_16 = tf.nn.relu(out_conv4)

    with tf.variable_scope("Block3_16"):
        with tf.variable_scope("conv5"):
            W_conv5 = weight_variable([3, 3, 16, 16], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv5)
            # b_conv5 = bias_variable(0., [16])
            conv5 = tf.nn.conv2d(out_block2_16, W_conv5, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv5
            BN5 = batch_norm(conv5, is_training)

        out_conv5 = tf.nn.relu(BN5)

        with tf.variable_scope("conv6"):
            W_conv6 = weight_variable([3, 3, 16, 16], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv6)
            # b_conv6 = bias_variable(0., [16])
            conv6 = tf.nn.conv2d(out_conv5, W_conv6, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv6
            BN6 = batch_norm(conv6, is_training)

        out_conv6 = tf.add(BN6, out_block2_16)
        out_block3_16 = tf.nn.relu(out_conv6)

    with tf.variable_scope("Block1_32"):
        with tf.variable_scope("conv32-1"):
            W_conv32_1 = weight_variable([3, 3, 16, 32], math.sqrt(2./(3*3*16)))
            tf.add_to_collection('w', W_conv32_1)
            # b_conv32_1 = bias_variable(0., [32])
            conv32_1 = tf.nn.conv2d(out_block3_16, W_conv32_1, strides=[1, 2, 2, 1],
                                 padding="SAME") #+ b_conv32_1
            BN32_1 = batch_norm(conv32_1, is_training)

        out_conv32_1 = tf.nn.relu(BN32_1)

        with tf.variable_scope("conv32-2"):
            W_conv32_2 = weight_variable([3, 3, 32, 32], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv32_2)
            # b_conv32_2 = bias_variable(0., [32])
            conv32_2 = tf.nn.conv2d(out_conv32_1, W_conv32_2, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv32_2
            BN32_2 = batch_norm(conv32_2, is_training)

        with tf.variable_scope("shortcut1"):
            #option B
            W_SC1 = weight_variable([1, 1, 16, 32], math.sqrt(2./(1*1*16)))
            tf.add_to_collection('w', W_SC1)
            # b_SC1 = bias_variable(0., [32])
            conv_SC1 = tf.nn.conv2d(out_block3_16, W_SC1, strides=[1, 2, 2, 1],
                                    padding="SAME") #+ b_SC1
            out_SC1 = batch_norm(conv_SC1, is_training)

            #option A
            # one = tf.ones(shape=[1, 1, ], dtype=tf.float32)
            # out = tf.nn.conv2d(out_block3_16, one, strides=[1, 2, 2, 1], padding="SAME")
            # padding_matrix = tf.zeros_like(out, tf.float32,name=None)
            # out_SC1 = tf.concat([out, padding_matrix], 3)

        out_conv32_2 = tf.add(BN32_2, out_SC1)
        out_block1_32 = tf.nn.relu(out_conv32_2)

    with tf.variable_scope("Block2_32"):
        with tf.variable_scope("conv32-3"):
            W_conv32_3 = weight_variable([3, 3, 32, 32], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv32_3)
            # b_conv32_3 = bias_variable(0., [32])
            conv32_3 = tf.nn.conv2d(out_block1_32, W_conv32_3, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv32_3
            BN32_3 = batch_norm(conv32_3, is_training)

        out_conv32_3 = tf.nn.relu(BN32_3)

        with tf.variable_scope("conv32-4"):
            W_conv32_4 = weight_variable([3, 3, 32, 32], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv32_4)
            # b_conv32_4 = bias_variable(0., [32])
            conv32_4 = tf.nn.conv2d(out_conv32_3, W_conv32_4, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv32_4
            BN32_4 = batch_norm(conv32_4, is_training)

        out_conv32_4 = tf.add(BN32_4, out_block1_32)
        out_block2_32 = tf.nn.relu(out_conv32_4)


    with tf.variable_scope("Block3_32"):
        with tf.name_scope("conv32-5"):
            W_conv32_5 = weight_variable([3, 3, 32, 32], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv32_5)
            # b_conv32_5 = bias_variable(0., [32])
            conv32_5 = tf.nn.conv2d(out_block2_32, W_conv32_5, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv32_5
            BN32_5 = batch_norm(conv32_5, is_training)

        out_conv32_5 = tf.nn.relu(BN32_5)

        with tf.variable_scope("conv32-6"):
            W_conv32_6 = weight_variable([3, 3, 32, 32], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv32_6)
            # b_conv32_6 = bias_variable(0., [32])
            conv32_6 = tf.nn.conv2d(out_conv32_5, W_conv32_6, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv32_6
            BN32_6 = batch_norm(conv32_6, is_training)

        out_conv32_6 = tf.add(BN32_6, out_block2_32)
        out_block3_32 = tf.nn.relu(out_conv32_6)

    with tf.variable_scope("Block1_64"):
        with tf.variable_scope("conv64-1"):
            W_conv64_1 = weight_variable([3, 3, 32, 64], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv64_1)
            # b_conv64_1 = bias_variable(0., [64])
            conv64_1 = tf.nn.conv2d(out_block3_32, W_conv64_1, strides=[1, 2, 2, 1],
                                 padding="SAME")# + b_conv64_1
            BN64_1 = batch_norm(conv64_1, is_training)

        out_conv64_1 = tf.nn.relu(BN64_1)

        with tf.variable_scope("conv64-2"):
            W_conv64_2 = weight_variable([3, 3, 64, 64], math.sqrt(2./(3*3*32)))
            tf.add_to_collection('w', W_conv64_2)
            # b_conv64_2 = bias_variable(0., [64])
            conv64_2 = tf.nn.conv2d(out_conv64_1, W_conv64_2, strides=[1, 1, 1, 1],
                                 padding="SAME")# + b_conv64_2
            BN64_2 = batch_norm(conv64_2, is_training)

        with tf.variable_scope("shortcut2"):
            #option B
            W_SC2 = weight_variable([1, 1, 32, 64], math.sqrt(2./(1*1*32)))
            tf.add_to_collection('w', W_SC2)
            # b_SC2 = bias_variable(0., [64])
            conv_SC2 = tf.nn.conv2d(out_block3_32, W_SC2, strides=[1, 2, 2, 1],
                                    padding="SAME") #+ b_SC2
            out_SC2 = batch_norm(conv_SC2, is_training)

            # option A
            # padding_matrix = tf.zeros_like(out_block3_32, tf.float32, name=None)
            # out_SC2 = tf.concat([out_block3_32, padding_matrix], 3)

        out_conv64_2 = tf.add(BN64_2, out_SC2)
        out_block1_64 = tf.nn.relu(out_conv64_2)

    with tf.variable_scope("Block2_64"):
        with tf.name_scope("conv64-3"):
            W_conv64_3 = weight_variable([3, 3, 64, 64], math.sqrt(2./(3*3*64)))
            tf.add_to_collection('w', W_conv64_3)
            # b_conv64_3 = bias_variable(0., [64])
            conv64_3 = tf.nn.conv2d(out_block1_64, W_conv64_3, strides=[1, 1, 1, 1],
                                    padding="SAME") #+ b_conv64_3
            BN64_3 = batch_norm(conv64_3, is_training)

        out_conv64_3 = tf.nn.relu(BN64_3)

        with tf.variable_scope("conv64-4"):
            W_conv64_4 = weight_variable([3, 3, 64, 64], math.sqrt(2./(3*3*64)))
            tf.add_to_collection('w', W_conv64_4)
            # b_conv64_4 = bias_variable(0., [64])
            conv64_4 = tf.nn.conv2d(out_conv64_3, W_conv64_4, strides=[1, 1, 1, 1],
                                    padding="SAME")# + b_conv64_4
            BN64_4 = batch_norm(conv64_4, is_training)

        out_conv64_4 = tf.add(BN64_4, out_block1_64)
        out_block2_64 = tf.nn.relu(out_conv64_4)

    with tf.variable_scope("Block3_64"):
        with tf.variable_scope("conv64-5"):
            W_conv64_5 = weight_variable([3, 3, 64, 64], math.sqrt(2./(3*3*64)))
            tf.add_to_collection('w', W_conv64_5)
            # b_conv64_5 = bias_variable(0., [64])
            conv64_5 = tf.nn.conv2d(out_block2_64, W_conv64_5, strides=[1, 1, 1, 1],
                                    padding="SAME") #+ b_conv64_5
            BN64_5 = batch_norm(conv64_5, is_training)

        out_conv64_5 = tf.nn.relu(BN64_5)

        with tf.variable_scope("conv64-6"):
            W_conv64_6 = weight_variable([3, 3, 64, 64], math.sqrt(2./(3*3*64)))
            tf.add_to_collection('w', W_conv64_6)
            # b_conv64_6 = bias_variable(0., [64])
            conv64_6 = tf.nn.conv2d(out_conv64_5, W_conv64_6, strides=[1, 1, 1, 1],
                                    padding="SAME")# + b_conv64_6
            BN64_6 = batch_norm(conv64_6, is_training)

        out_conv64_6 = tf.add(BN64_6, out_block2_64)
        out_block3_64 = tf.nn.relu(out_conv64_6)

    with tf.variable_scope('pool'):
        pooling = tf.nn.avg_pool(out_block3_64, ksize=[1, 8, 8, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("fc"):
        W_fc = weight_variable([64, 10], math.sqrt(2./(640)))
        tf.add_to_collection('w', W_fc)
        b_fc = bias_variable(0., [10])
        reshape = tf.reshape(pooling, [-1, 64])
        y_conv = tf.matmul(reshape, W_fc) + b_fc

        # y_conv = batch_norm(y_conv, is_training)
    # for v in tf.get_all_trainable_variables()
    return y_conv

def main():
    global Weight_decay
    is_train = True

    # img_train, label_train = read_and_decode("traindata.tfrecords", True)
    # img_test, label_test = read_and_decode("testdata.tfrecords", False)
    img_train, label_train = load_train_data()
    img_test, label_test = load_valid_data()
    img_train = (img_train - 128) / 128.0
    img_test = (img_test - 128) / 128.0

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_image')

    y = tf.placeholder(tf.int32, [None], name='label')
    is_training = tf.placeholder(tf.bool)

    y_pred = deep(x, is_training)

    with tf.name_scope('weight_loss'):
        weights_norm = tf.reduce_sum(
            input_tensor=Weight_decay * tf.stack(
                [tf.nn.l2_loss(i) for i in tf.get_collection('w')], axis=0),
            name='weights_norm'
        )

    with tf.name_scope('total_loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
        loss = cross_entropy + weights_norm

    with tf.name_scope('optimize'):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.piecewise_constant(global_step, [32000, 48000], [0.1, 0.01, 0.001])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # img_batch_train, label_batch_train = tf.train.shuffle_batch([img_train, label_train],
    #                                                             batch_size=128, capacity=50000,
    #                                                             min_after_dequeue=1000)
    # img_batch_test, label_batch_test = tf.train.shuffle_batch([img_test, label_test],
    #                                                           batch_size=200, capacity=20000,
    #                                                           min_after_dequeue=10000)

    def feed_dict(train, kk=1.0):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

        def get_batch(data, labels):
            id = np.random.randint(low=0, high=labels.shape[0], size=128, dtype=np.int32)
            return data[id, ...], labels[id]

        if train:
            tmp, ys = get_batch(img_train, label_train)
            xs = tmp
            tmp = np.pad(tmp, 4, 'constant')
            for ii in range(128):
                xx = np.random.randint(0, 9)
                yy = np.random.randint(0, 9)
                xs[ii, :] = np.fliplr(tmp[ii + 4, xx:xx + 32, yy:yy + 32, 4:7]) if np.random.randint(0, 2) == 1 \
                    else tmp[ii + 4, xx:xx + 32, yy:yy + 32, 4:7]
            k = kk
        else:
            # xs, ys = get_batch(valid_data, valid_labels)
            xs = img_test
            ys = label_test
            k = 1.0
        return {x: xs, y: ys, is_training: train}


    store_mean = tf.get_collection('store_mean')
    store_variance = tf.get_collection('store_variance')
    list_add = store_mean + store_variance
    list_add = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs", sess.graph)
        writer.flush()
        writer.close()

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(64000):
            # img, labels = sess.run([img_batch_train, label_batch_train])
            # for i in range(128):
            #     a = (img[i, :]*255+128).astype(np.uint8)
            #     plt.imshow(a)
            #     plt.show()
            if i % 100 == 0:
                # feed_dict = {x: img, y: labels, is_training: False}
                train_accuracy = accuracy.eval(feed_dict(False))
                # img_t, labels_t = sess.run([img_batch_test, label_batch_test])
                # feed_dict = {x: img_t, y: labels_t, is_training: False}
                test_accuracy = accuracy.eval(feed_dict(False))
                print('step %d, training accuracy %g , validation accuracy %g' % (i, train_accuracy, test_accuracy))
            # sess.run(train_step, feed_dict={x: img, y: labels, is_training: True})
            sess.run(train_step, feed_dict(True))
            # sess.run(list_add, feed_dict={x: img, is_training: True})
        coord.request_stop()
        coord.join(threads)

        img_batch_test, label_batch_test = tf.train.batch([img_test, label_test],
                                                                  batch_size=10000, capacity=50000)
        img_t, labels_t = sess.run([img_batch_test, label_batch_test])
        print('test accuracy %g' % accuracy.eval(feed_dict={x: img_t, y: labels_t, is_training:False}))

if __name__ == "__main__":
    main()