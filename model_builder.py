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

# Create a proper Keras model for TensorFlow 2.x
class TrafficSignModel(tf.keras.Model):
    def __init__(self, params):
        super(TrafficSignModel, self).__init__()
        self.params = params
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=params.conv1_d,
            kernel_size=params.conv1_k,
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=1-params.conv1_p)
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters=params.conv2_d,
            kernel_size=params.conv2_k,
            padding='same',
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.dropout2 = tf.keras.layers.Dropout(rate=1-params.conv2_p)
        
        self.conv3 = tf.keras.layers.Conv2D(
            filters=params.conv3_d,
            kernel_size=params.conv3_k,
            padding='same',
            activation='relu'
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.dropout3 = tf.keras.layers.Dropout(rate=1-params.conv3_p)
        
        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc4 = tf.keras.layers.Dense(params.fc4_size, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(rate=1-params.fc4_p)
        self.output_layer = tf.keras.layers.Dense(params.num_classes)
        
    def call(self, inputs, training=False):
        # First convolutional block
        x = self.conv1(inputs)
        x = self.pool1(x)
        if training:
            x = self.dropout1(x)
        x1 = tf.keras.layers.MaxPool2D(pool_size=4, strides=4, padding='same')(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.pool2(x)
        if training:
            x = self.dropout2(x)
        x2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.pool3(x)
        if training:
            x = self.dropout3(x)
        x3 = x
        
        # Flatten and concatenate multi-scale features
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)
        
        x = tf.concat([x1, x2, x3], axis=1)
        
        # Fully connected layers
        x = self.fc4(x)
        if training:
            x = self.dropout4(x)
        
        # Output layer
        return self.output_layer(x)

# Keep these functions for backward compatibility
def fully_connected(input, size):
    return tf.keras.layers.Dense(size)(input)

def fully_connected_relu(input, size):
    return tf.keras.layers.Dense(size, activation='relu')(input)

def conv_relu(input, kernel_size, depth):
    return tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=kernel_size,
        padding='same',
        activation='relu'
    )(input)

def pool(input, size):
    return tf.keras.layers.MaxPool2D(
        pool_size=size,
        strides=size,
        padding='same'
    )(input)

# This function is kept for backward compatibility with existing code
def model_pass(input, params, is_training):
    # Create a proper TrafficSignModel and call it
    model = TrafficSignModel(params)
    return model(input, training=is_training)