# model_builder.py
# This file defines the model architecture for the Traffic Sign Classification CNN.
# The architecture consists of:
#   - 3 Convolutional layers with ReLU activation
#   - 3 Max pooling layers
#   - Dropout after each pooling layer
#   - 1 Fully connected layer with ReLU activation
#   - 1 Output layer with softmax activation (implicit in the loss function)
#
# Note: The model architecture is designed to be consistent with the Keras implementation
# used in trainer.py and evaluator.py. Previously, this file contained a more complex architecture
# with multi-scale feature concatenation, but that has been replaced for consistency.

import tensorflow as tf
from collections import namedtuple
import time
import sys
import numpy as np

# ======== Model Parameters ========
Parameters = namedtuple('Parameters', [
    # Data parameters
    'num_classes', 'image_size',
    # Training parameters
    'batch_size', 'max_epochs', 'log_epoch', 'print_epoch',
    # Optimizations
    'learning_rate_decay', 'learning_rate',
    'l2_reg_enabled', 'l2_lambda',
    'early_stopping_enabled', 'early_stopping_patience',
    'resume_training',
    # Layers architecture
    'conv1_k', 'conv1_d', 'conv1_p',
    'conv2_k', 'conv2_d', 'conv2_p',
    'conv3_k', 'conv3_d', 'conv3_p',
    'fc4_size', 'fc4_p'
])

# ======== Utility Functions ========
def get_time_hhmmss(start = None):
    # Returns formatted timestamp or elapsed time if start is provided
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

def print_progress(iteration, total):
    # Prints progress bar to console
    str_format = "{0:.0f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(100 * iteration / float(total)))
    bar = '█' * filled_length + '-' * (100 - filled_length)
    sys.stdout.write('\r |%s| %s%%' % (bar, percents)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# ======== Model Components ========
def fully_connected(input, size):
    # Creates a fully connected layer with Xavier initialization
    initializer = tf.keras.initializers.GlorotUniform()  # TF2.x Xavier/Glorot initializer
    weights = tf.Variable(
        initializer(shape=[input.get_shape()[1], size]),
        name='weights'
    )
    biases = tf.Variable(
        tf.zeros([size]),
        name='biases'
    )
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    # Creates a fully connected layer with ReLU activation
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    # Creates a convolutional layer with ReLU activation
    input_depth = input.get_shape()[3]
    initializer = tf.keras.initializers.GlorotUniform()  # TF2.x Xavier/Glorot initializer
    weights = tf.Variable(
        initializer(shape=[kernel_size, kernel_size, input_depth, depth]),
        name='weights'
    )
    biases = tf.Variable(
        tf.zeros([depth]),
        name='biases'
    )
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    # Creates a max pooling layer
    return tf.nn.max_pool(
        input,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='SAME'
    )

def model_pass(input, params, is_training):
    """
    Creates a model architecture consistent with the Keras implementation used in trainer.py and evaluator.py.
    This ensures architectural consistency across all components of the system.
    
    Args:
        input: Input tensor
        params: Model parameters
        is_training: Boolean indicating if this is a training pass
        
    Returns:
        Output logits tensor
    """
    # Remove debug prints that can cause issues in graph mode
    # tf.print("Input shape:", tf.shape(input), output_stream=sys.stdout)
    
    # Handle is_training as either a Python bool or a TF tensor
    # If it's a tensor, this will safely use it in graph mode
    # If it's a Python bool, it will just use the value directly
    def maybe_dropout(x, keep_rate):
        if isinstance(is_training, bool):
            # Python bool case
            if is_training:
                return tf.nn.dropout(x, rate=1-keep_rate)
            return x
        else:
            # TF tensor case
            return tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(x, rate=1-keep_rate),
                lambda: x
            )
    
    with tf.name_scope('conv1'):
        conv1 = conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
    
    with tf.name_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = maybe_dropout(pool1, params.conv1_p)
    
    with tf.name_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params.conv2_k, depth=params.conv2_d)
    
    with tf.name_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = maybe_dropout(pool2, params.conv2_p)
    
    with tf.name_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params.conv3_k, depth=params.conv3_d)
    
    with tf.name_scope('pool3'):
        pool3 = pool(conv3, size=2)
        pool3 = maybe_dropout(pool3, params.conv3_p)
    
    # Flatten layer (simplified to match Keras implementation)
    shape = pool3.get_shape().as_list()
    flattened = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    
    with tf.name_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params.fc4_size)
        fc4 = maybe_dropout(fc4, params.fc4_p)
    
    with tf.name_scope('out'):
        logits = fully_connected(fc4, size=params.num_classes)
    
    return logits