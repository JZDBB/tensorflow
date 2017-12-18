import tensorflow as tf

def weight_variable(shape):
  initial = tf.random_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(x, shape):
    initial = tf.constant(x, shape=shape)
    return tf.Variable(initial)

def batch_norm(x):
    return x

def conv(x, Weight_shape, bias_val, bias_shape, stride):
    out = 1
    return out


def deep(x):
    with tf.name_scope("reshape"):
        x_orig = tf.reshape(x, [-1, 32, 32, 3])

    with tf.name_scope("pre_conv"):
        with tf.name_scope("conv"):
            W_preconv = weight_variable([3, 3, 3, 16])
            b_preconv = bias_variable(0, [16])
            preconv = tf.nn.conv2d(x_orig, W_preconv, strides=[1, 1, 1, 1],
                                   padding="SAME") + b_preconv
            preBN = batch_norm(preconv)

        out_preconv = tf.nn.relu(preBN)

    with tf.name_scope("Block1_16"):
        with tf.name_scope("conv1"):
            W_conv1 = weight_variable([3, 3, 16, 16])
            b_conv1 = bias_variable(0, [16])
            conv1 = tf.nn.conv2d(out_preconv, W_conv1, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv1
            BN1 = batch_norm(conv1)

        out_conv1 = tf.nn.relu(BN1)

        with tf.name_scope("conv2"):
            W_conv2 = weight_variable([3, 3, 16, 16])
            b_conv2 = bias_variable(0, [16])
            conv2 = tf.nn.conv2d(out_conv1, W_conv2, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv2
            BN2 = batch_norm(conv2)

        out_conv2 = tf.add(BN2, x_orig)
        out_block1_16 = tf.nn.relu(out_conv2)

    with tf.name_scope("Block2_16"):
        with tf.name_scope("conv3"):
            W_conv3 = weight_variable([3, 3, 16, 16])
            b_conv3 = bias_variable(0, [16])
            conv3 = tf.nn.conv2d(out_block1_16, W_conv3, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv3
            BN3 = batch_norm(conv3)

        out_conv3 = tf.nn.relu(BN3)

        with tf.name_scope("conv4"):
            W_conv4 = weight_variable([3, 3, 16, 16])
            b_conv4 = bias_variable(0, [16])
            conv4 = tf.nn.conv2d(out_conv3, W_conv4, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv4
            BN4 = batch_norm(conv4)

        out_conv4 = tf.add(BN4, out_block1_16)
        out_block2_16 = tf.nn.relu(out_conv4)

    with tf.name_scope("Block3_16"):
        with tf.name_scope("conv5"):
            W_conv5 = weight_variable([3, 3, 16, 16])
            b_conv5 = bias_variable(0, [16])
            conv5 = tf.nn.conv2d(out_block2_16, W_conv5, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv5
            BN5 = batch_norm(conv5)

        out_conv5 = tf.nn.relu(BN5)

        with tf.name_scope("conv6"):
            W_conv6 = weight_variable([3, 3, 16, 16])
            b_conv6 = bias_variable(0, [16])
            conv6 = tf.nn.conv2d(out_conv5, W_conv6, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv6
            BN6 = batch_norm(conv6)

        out_conv6 = tf.add(BN6, out_block2_16)
        out_block3_16 = tf.nn.relu(out_conv6)

    with tf.name_scope("Block1_32"):
        with tf.name_scope("conv32-1"):
            W_conv32_1 = weight_variable([3, 3, 16, 32])
            b_conv32_1 = bias_variable(0, [32])
            conv32_1 = tf.nn.conv2d(out_block3_16, W_conv32_1, strides=[1, 2, 2, 1],
                                 padding="SAME") + b_conv32_1
            BN32_1 = batch_norm(conv32_1)

        out_conv32_1 = tf.nn.relu(BN32_1)

        with tf.name_scope("conv32-2"):
            W_conv32_2 = weight_variable([3, 3, 32, 32])
            b_conv32_2 = bias_variable(0, [32])
            conv32_2 = tf.nn.conv2d(out_conv32_1, W_conv32_2, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_2
            BN32_2 = batch_norm(conv32_2)

        with tf.name_scope("shortcut1"):
            W_SC1 = weight_variable([1, 1, 16, 32])
            b_SC1 = bias_variable(0, [32])
            conv_SC1 = tf.nn.conv2d(out_block3_16, W_SC1, strides=[1, 2, 2, 1],
                                    padding="SAME") + b_SC1
            out_SC1 = batch_norm(conv_SC1)

        out_conv32_2 = tf.add(BN32_2, out_SC1)
        out_block1_32 = tf.nn.relu(out_conv32_2)

    with tf.name_scope("Block2_32"):
        with tf.name_scope("conv32-3"):
            W_conv32_3 = weight_variable([3, 3, 16, 32])
            b_conv32_3 = bias_variable(0, [32])
            conv32_3 = tf.nn.conv2d(out_block1_32, W_conv32_3, strides=[1, 2, 2, 1],
                                 padding="SAME") + b_conv32_3
            BN32_3 = batch_norm(conv32_3)

        out_conv32_3 = tf.nn.relu(BN32_3)

        with tf.name_scope("conv32-4"):
            W_conv32_4 = weight_variable([3, 3, 32, 32])
            b_conv32_4 = bias_variable(0, [32])
            conv32_4 = tf.nn.conv2d(out_conv32_3, W_conv32_4, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_4
            BN32_4 = batch_norm(conv32_4)

        out_conv32_4 = tf.add(BN32_4, out_block1_32)
        out_block2_32 = tf.nn.relu(out_conv32_4)


    with tf.name_scope("Block3_32"):
        with tf.name_scope("conv32-5"):
            W_conv32_5 = weight_variable([3, 3, 16, 32])
            b_conv32_5 = bias_variable(0, [32])
            conv32_5 = tf.nn.conv2d(out_block2_32, W_conv32_5, strides=[1, 2, 2, 1],
                                 padding="SAME") + b_conv32_5
            BN32_5 = batch_norm(conv32_5)

        out_conv32_5 = tf.nn.relu(BN32_5)

        with tf.name_scope("conv32-6"):
            W_conv32_6 = weight_variable([3, 3, 32, 32])
            b_conv32_6 = bias_variable(0, [32])
            conv32_6 = tf.nn.conv2d(out_conv32_5, W_conv32_6, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_6
            BN32_6 = batch_norm(conv32_6)

        out_conv32_6 = tf.add(BN32_6, out_block2_32)
        out_block3_32 = tf.nn.relu(out_conv32_6)

    with tf.name_scope("Block1_64"):
        with tf.name_scope("conv64-1"):
            W_conv64_1 = weight_variable([3, 3, 32, 64])
            b_conv64_1 = bias_variable(0, [64])
            conv64_1 = tf.nn.conv2d(out_block3_32, W_conv64_1, strides=[1, 2, 2, 1],
                                 padding="SAME") + b_conv64_1
            BN64_1 = batch_norm(conv64_1)

        out_conv64_1 = tf.nn.relu(BN64_1)

        with tf.name_scope("conv64-2"):
            W_conv64_2 = weight_variable([3, 3, 64, 64])
            b_conv64_2 = bias_variable(0, [64])
            conv64_2 = tf.nn.conv2d(out_conv64_1, W_conv32_2, strides=[1, 1, 1, 1],
                                 padding="SAME") + b_conv32_2
            BN32_2 = batch_norm(conv32_2)

        with tf.name_scope("shortcut"):
            W_SC1 = weight_variable([1, 1, 16, 32])
            b_SC1 = bias_variable(0, [32])
            conv_SC1 = tf.nn.conv2d(out_block3_16, W_SC1, strides=[1, 2, 2, 1],
                                    padding="SAME") + b_SC1
            out_SC1 = batch_norm(conv_SC1)

        out_conv32_2 = tf.add(BN32_2, out_SC1)
        out_block1_32 = tf.nn.relu(out_conv32_2)