import tensorflow as tf

def weight_variable(shape):
  initial = tf.random_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(x, shape):
    initial = tf.constant(x, shape=shape)
    return tf.Variable(initial)


def ZF_Net(x):
    with tf.name_scope("reshape"):
        x_image = tf.reshape(x, [-1, 224, 224, 3])

    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([7, 7, 3, 96])
        b_conv1 = bias_variable(0., [96])
        conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1],
                             padding="SAME") + b_conv1
        out_conv1 = tf.nn.relu(conv1)

    with tf.name_scope("pool1"):
        out_pool1 = tf.nn.max_pool(out_conv1, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding="SAME")

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 96, 256])
        b_conv2 = bias_variable(1., [256])
        conv2 = tf.nn.conv2d(out_pool1, W_conv2, strides=[1, 2, 2, 1],
                             padding="VALID") + b_conv2
        out_conv2 = tf.nn.relu(conv2)

    with tf.name_scope("pool2"):
        out_pool2 = tf.nn.max_pool(out_conv2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding="SAME")

    with tf.name_scope("conv3"):
        W_conv3 = weight_variable([3, 3, 256, 384])
        b_conv3 = bias_variable(0., [384])
        conv3 = tf.nn.conv2d(out_pool2, W_conv3, strides=[1, 1, 1, 1],
                             padding="SAME") + b_conv3
        out_conv3 = tf.nn.relu(conv3)

    with tf.name_scope("conv4"):
        W_conv4 = weight_variable([3, 3, 384, 384])
        b_conv4 = bias_variable(1., [384])
        conv4 = tf.nn.conv2d(out_conv3, W_conv4, strides=[1, 1, 1, 1],
                             padding="SAME") + b_conv4
        out_conv4 = tf.nn.relu(conv4)

    with tf.name_scope("conv5"):
        W_conv5 = weight_variable([3, 3, 384, 256])
        b_conv5 = bias_variable(1., [256])
        conv5 = tf.nn.conv2d(conv4, W_conv5, strides=[1, 1, 1, 1],
                             padding="SAME") + b_conv5
        out_conv5 = tf.nn.relu(conv5)


    with tf.name_scope("pool5"):
        out_pool5 = tf.nn.max_pool(out_conv5, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding="VALID")

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([6*6*256, 4096])
        b_fc1 = bias_variable(1., [4096])
        out_pool5_flat = tf.reshape(out_pool5, [-1, 6*6*256])
        out_fc1 = tf.nn.relu(tf.matmul(out_pool5_flat, W_fc1) + b_fc1)

    with tf.name_scope("dropout1"):
        # keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(out_fc1, 0.5)

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([4096, 4096])
        b_fc2 = bias_variable(1., [4096])
        out_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope("dropout2"):
        # keep_prob = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(out_fc2, 0.5)

    with tf.name_scope("fc3"):
        W_fc3 = weight_variable([4096, 1000])
        b_fc3 = bias_variable(1., [1000])
        out_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    return out_fc3


def main():
    x = tf.placeholder(tf.float32, [None, 224*224*3], name='input_image')
    y = tf.placeholder(tf.int32, [None], name='label')

    y_pred = ZF_Net(x)

    print(y.shape, y_pred.shape)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
        # important!!!
        # difference
        # sparse_softmax_cross_entropy_with_logits(do not need onehot encoding) and
        # softmax_cross_entropy_with_logits(need preprocess use onehot encoding)

    with tf.name_scope('optimize'):
        # tips~~~
        #change learning rate according to training times
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
        # sess.run(tf.global_variables_initializer())


if __name__ == "__main__":
    main()