import tensorflow as tf

tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model.')
tf.app.flags.DEFINE_integer('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_string('data_path', './data', 'the path for data.')
FLAGS = tf.app.flags.FLAGS




