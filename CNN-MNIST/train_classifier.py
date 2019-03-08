from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from net_structure import net
from input_function import input
import os
import train_evaluation

# FLAGS
tf.app.flags.DEFINE_string(
    'train_dir', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint', 10, 'Maximum number of checkpoints that TensorFlow will keep.')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of epochs for training.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')
tf.app.flags.DEFINE_boolean('is_training', False, 'Training/Testing.')
tf.app.flags.DEFINE_boolean('fine_tuning', False, 'Fine tuning is desired or not?.')
tf.app.flags.DEFINE_boolean('online_test', True, 'Fine tuning is desired or not?.')
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Automatically put the variables on CPU if there is no GPU support.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Demonstrate which variables are on what device.')
FLAGS = tf.app.flags.FLAGS

# deal errors
if not os.path.isabs(FLAGS.train_dir):
    raise ValueError('You must assign absolute path for --train_dir')

if not os.path.isabs(FLAGS.checkpoint_dir):
    raise ValueError('You must assign absolute path for --checkpoint_dir')

# input images
mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=False)
data = input.provide_data(mnist)
num_train_samples, height, width, num_channels = data.train.images.shape

# draw graph
graph = tf.Graph()
with graph.as_default():
    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # learning rate policy
    decay_steps = int(num_train_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
    label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
    dropout_param = tf.placeholder(tf.float32)

    # MODEL
    arg_scope = net.net_arg_scope(weight_decay=0.0005, is_training=FLAGS.is_training)
    with tf.contrib.framework.arg_scope(arg_scope):
        logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes,
                                                  dropout_keep_prob=dropout_param,
                                                  is_training=FLAGS.is_training)

    # Define loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))

    # Accuracy
    with tf.name_scope('accuracy'):
        # Evaluate the model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))

        # Accuracy calculation
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Define optimizer by its default values
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 'train_op' is a operation that is run for gradient update on parameters.
    # Each execution of 'train_op' is a training step.
    # By passing 'global_step' to the optimizer, each time that the 'train_op' is run, Tensorflow
    # update the 'global_step' and increment it by one!

    # gradient update.
    with tf.name_scope('train'):
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    ###############################################
    ############ Define Sammaries #################
    ###############################################

    # Image summaries(draw three random images from data in both training and testing phases)
    # The image summaries is only created for train summaries and it get three random images from the training set.
    arr = np.random.randint(data.train.images.shape[0], size=(3,))
    tf.summary.image('images', data.train.images[arr], max_outputs=3,
                     collections=['per_epoch_train'])

    # Histogram and scalar summaries summaries
    # sparsity: This summary is the fraction of zero activation for the output of each layer!
    # activations: This summary is the histogram of activation for the output of each layer!
    # WARNING: tf.summary.histogram can be very time consuming so it will be calculated per epoch!
    for end_point in end_points:
        x = end_points[end_point]
        tf.summary.scalar('sparsity/' + end_point,
                          tf.nn.zero_fraction(x), collections=['train', 'test'])
        tf.summary.histogram('activations/' + end_point, x, collections=['per_epoch_train'])

    # Summaries for loss and accuracy
    tf.summary.scalar("loss", loss, collections=['train', 'test'])
    tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
    tf.summary.scalar("global_step", global_step, collections=['train'])
    tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

    # Merge all summaries together.
    summary_train_op = tf.summary.merge_all('train')
    summary_test_op = tf.summary.merge_all('test')
    summary_epoch_train_op = tf.summary.merge_all('per_epoch_train')

    ########################################################
    ############ # Defining the tensors list ###############
    ########################################################

    tensors_key = ['cost', 'accuracy', 'train_op', 'global_step', 'image_place', 'label_place', 'dropout_param',
                   'summary_train_op', 'summary_test_op', 'summary_epoch_train_op']
    tensors = [loss, accuracy, train_op, global_step, image_place, label_place, dropout_param, summary_train_op,
               summary_test_op, summary_epoch_train_op]
    tensors_dictionary = dict(zip(tensors_key, tensors))

    ############################################
    ############ Run the Session ###############
    ############################################
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf)

    with sess.as_default():
        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)

        sess.run(tf.global_variables_initializer())

        ###################################################
        ############ Training / Evaluation ###############
        ###################################################
        train_evaluation.train(sess=sess, saver=saver, tensors=tensors_dictionary, data=data,
                               train_dir=FLAGS.train_dir,
                               finetuning=FLAGS.fine_tuning, online_test=FLAGS.online_test,
                               num_epochs=FLAGS.num_epochs, checkpoint_dir=FLAGS.checkpoint_dir,
                               batch_size=FLAGS.batch_size)

        # Test in the end of experiment.
        train_evaluation.evaluation(sess=sess, saver=saver, tensors=tensors_dictionary, data=data,
                                    checkpoint_dir=FLAGS.checkpoint_dir)
