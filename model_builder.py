# model_builder.py
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
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

def print_progress(iteration, total):
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
    weights = tf.get_variable( 'weights',
                              shape = [input.get_shape()[1], size],
                              initializer = tf.contrib.layers.xavier_initializer()
    )
    biases = tf.get_variable( 'biases',
                             shape = [size],
                             initializer = tf.constant_initializer(0.0)
    )
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    weights = tf.get_variable( 'weights',
                              shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
                              initializer = tf.contrib.layers.xavier_initializer()
    )
    biases = tf.get_variable( 'biases',
                             shape = [depth],
                             initializer = tf.constant_initializer(0.0)
    )
    conv = tf.nn.conv2d(input, weights, strides = [1, 1, 1, 1], padding = 'SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    return tf.nn.max_pool(
        input,
        ksize = [1, size, size, 1],
        strides = [1, size, size, 1],
        padding = 'SAME'
    )

def model_pass(input, params, is_training):
    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size = params.conv1_k, depth = params.conv1_d)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size = 2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob = params.conv1_p), lambda: pool1)
    
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size = params.conv2_k, depth = params.conv2_d)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size = 2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob = params.conv2_p), lambda: pool2)
    
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size = params.conv3_k, depth = params.conv3_d)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size = 2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob = params.conv3_p), lambda: pool3)
    
    # Fully connected
    pool1 = pool(pool1, size = 4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])
    
    pool2 = pool(pool2, size = 2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    
    flattened = tf.concat([pool1, pool2, pool3], 1)
    
    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size = params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob = params.fc4_p), lambda: fc4)
    
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size = params.num_classes)
    
    return logits