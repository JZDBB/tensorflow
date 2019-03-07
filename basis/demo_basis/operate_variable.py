from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.python.framework import ops

# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# The user is prompted to input an absolute path.
# os.path.expanduser is leveraged to transform '~' sign to the corresponding path indicator.
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

# Create three variables with some default values.
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1),
                      name="weights")
biases = tf.Variable(tf.zeros([3]), name="biases")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

# Get all the variables' tensors and store them in a list.
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

############################################
######## Customized initializer ############
############################################

## Initialation of some custom variables.
## In this part we choose some variables and only initialize them rather than initializing all variables.

# "variable_list_custom" is the list of variables that we want to initialize.
variable_list_custom = [weights, custom_variable]

# The initializer
init_custom_op = tf.variables_initializer(var_list=variable_list_custom )


########################################
######## Global initializer ############
########################################

# Method-1
# Add an op to initialize the variables.
init_all_op = tf.global_variables_initializer()

# Method-2
# init_all_op = tf.variables_initializer(var_list=all_variables_list)

##########################################################
######## Initialization using other variables ############
##########################################################

# Create another variable with the same value as 'weights'.
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

# Now, the variable must be initialized.
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# Some basic operations
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("a + b =", sess.run(x))
    print("a/b =", sess.run(y))
    # Run the initializer operation.
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_WeightsNew_op)

# Closing the writer.
writer.close()
sess.close()