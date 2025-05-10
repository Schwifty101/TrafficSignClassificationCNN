# evaluator.py
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
        X: Input images
        k: Number of top predictions to return
        
    Returns:
        Tuple of (values, indices) for the top-k predictions
    """
    print_debug(f"Getting top-{k} predictions for {len(X)} images")
    start_time = time.time()
    
    paths = Paths(params)
    
    # Check if the model file exists
    model_file = paths.model_path
    if not os.path.exists(model_file):
        print_debug(f"Model file not found at {model_file}")
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    # Set up TF2 strategy for potential GPU usage
    print_debug("Checking for available GPUs...")
    gpus = tf.config.list_physical_devices('GPU')
    print_debug(f"Found {len(gpus)} GPUs")
    
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 0 else tf.distribute.get_strategy()
    print_debug(f"Using {'multi-GPU' if len(gpus) > 0 else 'CPU'} strategy")
    
    with strategy.scope():
        # Define model architecture using tf.keras (same as in trainer.py)
        print_debug("Building model architecture...")
        # This is the standard model architecture used throughout the project
        # Model Architecture:
        # - 3 Convolutional layers with ReLU activation
        # - 3 Max pooling layers with dropout
        # - Flatten layer
        # - 1 Fully connected layer with ReLU activation and dropout
        # - Output layer with softmax activation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(params.image_size[0], params.image_size[1], 1)))
        
        # Layer 1: Convolutional + MaxPooling
        model.add(tf.keras.layers.Conv2D(params.conv1_d, (params.conv1_k, params.conv1_k), activation='relu', padding='same', 
                                         name='conv1'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(tf.keras.layers.Dropout(1 - params.conv1_p, name='dropout1'))
        
        # Layer 2: Convolutional + MaxPooling
        model.add(tf.keras.layers.Conv2D(params.conv2_d, (params.conv2_k, params.conv2_k), activation='relu', padding='same',
                                         name='conv2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
        model.add(tf.keras.layers.Dropout(1 - params.conv2_p, name='dropout2'))
        
        # Layer 3: Convolutional + MaxPooling
        model.add(tf.keras.layers.Conv2D(params.conv3_d, (params.conv3_k, params.conv3_k), activation='relu', padding='same',
                                         name='conv3'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3'))
        model.add(tf.keras.layers.Dropout(1 - params.conv3_p, name='dropout3'))
        
        # Flatten and fully connected layers
        model.add(tf.keras.layers.Flatten(name='flatten'))
        model.add(tf.keras.layers.Dense(params.fc4_size, activation='relu', 
                                        kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                        name='fc4'))
        model.add(tf.keras.layers.Dropout(1 - params.fc4_p, name='dropout4'))
        model.add(tf.keras.layers.Dense(params.num_classes, activation='softmax', name='output'))
        
        # Compile the model - not strictly necessary for inference but good practice
        print_debug("Compiling model...")
        model.compile(
            optimizer='adam',  # Doesn't matter for inference
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Load weights
    print_debug(f"Loading model weights from {paths.model_path}")
    try:
        model.load_weights(paths.model_path)
        print_debug("Model weights loaded successfully")
    except Exception as e:
        print_debug(f"Error loading model weights: {e}")
        raise
    
    # Set batch size for prediction
    batch_size = 32
    num_batches = int(np.ceil(len(X) / batch_size))
    print_debug(f"Processing predictions in {num_batches} batches of size {batch_size}")
    
    # Make predictions in batches
    all_predictions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch_X = X[start_idx:end_idx]
        
        # Ensure batch is float32 for consistent precision
        if batch_X.dtype != np.float32:
            print_debug(f"Converting batch from {batch_X.dtype} to float32")
            batch_X = batch_X.astype(np.float32)
        
        # Check if the data range is appropriate
        data_min, data_max = batch_X.min(), batch_X.max()
        print_debug(f"Batch {i+1} data range: [{data_min:.6f}, {data_max:.6f}]")
        
        print_debug(f"Processing batch {i+1}/{num_batches} with {len(batch_X)} images")
        predictions = model.predict(batch_X, verbose=0)
        all_predictions.append(predictions)
        
        # Update progress bar
        progress_bar(i+1, num_batches, 
                     prefix=f"Prediction progress", 
                     suffix=f"Batch {i+1}/{num_batches}")
    
    # Combine all batches
    all_predictions = np.vstack(all_predictions)
    print_debug(f"All predictions completed with shape {all_predictions.shape}")
    
    # Get top-k values and indices
    print_debug(f"Extracting top-{k} predictions from {all_predictions.shape}")
    # Initialize with proper shape - k rows, len(X) columns
    top_values = np.zeros((k, len(X)))
    top_indices = np.zeros((k, len(X)), dtype=np.int32)
    
    print_debug(f"top_values shape: {top_values.shape}, top_indices shape: {top_indices.shape}")
    
    for i in range(len(X)):
        # Get the top k predictions for this sample
        indices = np.argsort(all_predictions[i])[-k:][::-1]
        values = all_predictions[i][indices]
        
        # Debug output for troubleshooting
        if i < 3:  # Only print for first few samples to avoid log flooding
            print_debug(f"Sample {i}: indices shape={indices.shape}, values shape={values.shape}")
            print_debug(f"Target slice shape: top_values[:, {i}].shape = {top_values[:, i].shape}")
        
        # Make sure we're handling the right shapes and fix broadcasting issue
        if len(indices) != k:
            # Handle case where we have fewer than k classes
            indices = np.pad(indices, (0, k - len(indices)), 'constant')
            values = np.pad(values, (0, k - len(values)), 'constant')
        
        # Reshape values and indices to ensure they have the right dimensions for assignment
        values_reshaped = np.reshape(values, (k,))
        indices_reshaped = np.reshape(indices, (k,))
            
        # Assign values to the correct slice with proper shape matching
        top_values[:, i] = values_reshaped
        top_indices[:, i] = indices_reshaped
    
    print_debug(f"Top-{k} predictions extracted")
    print_debug(f"Prediction completed in {time.time() - start_time:.2f}s")
    
    # Return (values, indices) as was done in the original code
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
        # Use .weights.h5 extension for compatibility with Keras 3
        return self.root_path + "model.weights.h5"

if __name__ == "__main__":
    # Test code
    print_debug("Testing evaluator module")
    try:
        # Load some test data
        from data_loader import load_pickled_data
        import numpy as np
        X_test, filenames = load_pickled_data('data/test.p', ['features', 'filenames'])
        # Create dummy test labels for code compatibility
        y_test = np.zeros(X_test.shape[0], dtype=np.int32)
        print_debug(f"Loaded test data with {len(X_test)} samples (dummy labels created)")
        
        # Create parameters
        params = Parameters(
            num_classes=43,
            image_size=(32, 32),
            batch_size=256,
            max_epochs=1,
            log_epoch=1,
            print_epoch=1,
            learning_rate_decay=False,
            learning_rate=0.0001,
            l2_reg_enabled=True,
            l2_lambda=0.0001,
            early_stopping_enabled=True,
            early_stopping_patience=100,
            resume_training=True,
            conv1_k=5, conv1_d=32, conv1_p=0.9,
            conv2_k=5, conv2_d=64, conv2_p=0.8,
            conv3_k=5, conv3_d=128, conv3_p=0.7,
            fc4_size=1024, fc4_p=0.5
        )
        
        # Preprocess data if needed
        from data_processor import preprocess_dataset
        print_debug(f"Before preprocessing: dtype={X_test.dtype}, min={X_test.min()}, max={X_test.max()}")
        X_test_processed, y_test_processed = preprocess_dataset(X_test, y_test)
        print_debug(f"After preprocessing: dtype={X_test_processed.dtype}, min={X_test_processed.min():.6f}, max={X_test_processed.max():.6f}")
        
        # Get top predictions
        print_debug("Getting top predictions")
        top_preds = get_top_k_predictions(params, X_test_processed[:10], k=5)
        
        # Print some results
        print_debug("Top predictions for first 5 samples:")
        for i in range(5):
            print_debug(f"Sample {i}: {top_preds[1][:, i]} with confidences {top_preds[0][:, i]}")
            
    except Exception as e:
        print_debug(f"Test failed: {e}")
        import traceback
        traceback.print_exc()