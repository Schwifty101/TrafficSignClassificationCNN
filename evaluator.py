import os
import tensorflow as tf
import numpy as np
import time
import sys
from model_builder import Parameters

def print_debug(message):
    """Helper function to print debug messages with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {message}")

def progress_bar(current, total, bar_length=50, prefix='', suffix=''):
    """Display a progress bar in the terminal"""
    filled_length = int(round(bar_length * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percents}% {suffix}')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def get_top_k_predictions(params, X, k=5):
    """
    Get top-k predictions for input images using the TensorFlow 2.x model
    
    Args:
        params: Model parameters
        X: Input images (expected shape: (num_samples, 32, 32, 1), range: [0, 1])
        k: Number of top predictions to return
        
    Returns:
        Tuple of (values, indices) for the top-k predictions
    """
    print_debug(f"Getting top-{k} predictions for {len(X)} images")
    start_time = time.time()
    
    paths = Paths(params)
    model_file = paths.model_path
    if not os.path.exists(model_file):
        print_debug(f"Model file not found at {model_file}")
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    # Validate input shape and normalize
    print_debug(f"Input X shape: {X.shape}, dtype: {X.dtype}")
    if len(X.shape) != 4 or X.shape[1:3] != (32, 32) or X.shape[3] != 1:
        print_debug("Invalid input shape, expected (num_samples, 32, 32, 1)")
        if len(X.shape) == 3 and X.shape[1:3] == (32, 32):
            print_debug("Adding channel dimension")
            X = X.reshape(X.shape + (1,))
        elif X.shape[3] == 3:
            print_debug("Converting RGB to grayscale")
            X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
            X = X.reshape(X.shape[0], 32, 32, 1)
        else:
            raise ValueError(f"Cannot reshape X with shape {X.shape} to (num_samples, 32, 32, 1)")
    
    # Ensure normalization
    X_min, X_max = X.min(), X.max()
    print_debug(f"Input X range: [{X_min:.6f}, {X_max:.6f}]")
    if X_max > 1.0 or X_min < 0.0:
        print_debug("Normalizing X to [0, 1]")
        X = (X / 255.0).astype(np.float32) if X_max > 1.0 else X.astype(np.float32)
    
    # Set up strategy
    gpus = tf.config.list_physical_devices('GPU')
    print_debug(f"Found {len(gpus)} GPUs")
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 0 else tf.distribute.get_strategy()
    
    with strategy.scope():
        # Define model (must match trainer.py)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(params.conv1_d, (params.conv1_k, params.conv1_k), activation='relu', padding='same', name='conv1'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
            tf.keras.layers.Dropout(1 - params.conv1_p, name='dropout1'),
            tf.keras.layers.Conv2D(params.conv2_d, (params.conv2_k, params.conv2_k), activation='relu', padding='same', name='conv2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
            tf.keras.layers.Dropout(1 - params.conv2_p, name='dropout2'),
            tf.keras.layers.Conv2D(params.conv3_d, (params.conv3_k, params.conv3_k), activation='relu', padding='same', name='conv3'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
            tf.keras.layers.Dropout(1 - params.conv3_p, name='dropout3'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(params.fc4_size, activation='relu', 
                                 kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                 name='fc4'),
            tf.keras.layers.Dropout(1 - params.fc4_p, name='dropout4'),
            tf.keras.layers.Dense(params.num_classes, activation='softmax', name='output')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load weights
    print_debug(f"Loading weights from {model_file}")
    try:
        model.load_weights(model_file)
    except Exception as e:
        print_debug(f"Error loading weights: {e}")
        raise
    
    # Predict in batches
    batch_size = 32
    num_batches = int(np.ceil(len(X) / batch_size))
    all_predictions = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch_X = X[start_idx:end_idx].astype(np.float32)
        predictions = model.predict(batch_X, verbose=0)
        all_predictions.append(predictions)
        progress_bar(i + 1, num_batches, prefix="Prediction progress", suffix=f"Batch {i+1}/{num_batches}")
    
    all_predictions = np.vstack(all_predictions)
    print_debug(f"All predictions shape: {all_predictions.shape}")
    
    # Extract top-k
    top_values = np.zeros((k, len(X)))
    top_indices = np.zeros((k, len(X)), dtype=np.int32)
    for i in range(len(X)):
        indices = np.argsort(all_predictions[i])[-k:][::-1]
        values = all_predictions[i][indices]
        top_values[:, i] = values
        top_indices[:, i] = indices
    
    print_debug(f"Top-{k} predictions extracted in {time.time() - start_time:.2f}s")
    return (top_values, top_indices)

class Paths:
    def __init__(self, params):
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        
    def get_model_name(self, params):
        model_name = f"k{params.conv1_k}d{params.conv1_d}p{params.conv1_p}_k{params.conv2_k}d{params.conv2_d}p{params.conv2_p}_k{params.conv3_k}d{params.conv3_d}p{params.conv3_p}_fc{params.fc4_size}p{params.fc4_p}"
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name
    
    def get_variables_scope(self, params):
        var_scope = f"k{params.conv1_k}d{params.conv1_d}_k{params.conv2_k}d{params.conv2_d}_k{params.conv3_k}d{params.conv3_d}_fc{params.fc4_size}_fc0"
        return var_scope
    
    def get_model_path(self):
        return self.root_path + "model.weights.h5"