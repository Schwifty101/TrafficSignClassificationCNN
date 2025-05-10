# predictor.py
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_builder import Parameters, model_pass
from data_loader import load_sign_names
from data_processor import preprocess_dataset
from skimage import io

def predict_on_new_images(params, image_paths):
    """
    Predict traffic signs on new images
    Args:
        params: model parameters
        image_paths: paths to new images
    """
    sign_names = load_sign_names()
    
    # Initialize with float32 to maintain precision throughout the pipeline
    X_custom = np.empty([0, 32, 32, 3], dtype=np.float32)
    
    # Debug point: Loading custom images
    print(f"Loading {len(image_paths)} custom images for prediction")
    
    for i in range(len(image_paths)):
        # Read image
        image = io.imread(image_paths[i])
        
        # Resize image to match the expected input size
        from skimage.transform import resize
        # Resize to match the expected input dimensions (32x32)
        if image.shape[0] != 32 or image.shape[1] != 32:
            print(f"Resizing image from {image.shape} to (32, 32, 3)")
            image = resize(image, (32, 32, 3), anti_aliasing=True)
            # Ensure we're in the right data range after resize
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
        
        # Convert to float32 immediately
        image = image.astype(np.float32)
        X_custom = np.append(X_custom, [image[:, :, :3]], axis=0)
    
    # Preprocess the custom images
    print(f"Before preprocessing: dtype={X_custom.dtype}, min={X_custom.min()}, max={X_custom.max()}")
    X_custom, _ = preprocess_dataset(X_custom)
    print(f"After preprocessing: dtype={X_custom.dtype}, min={X_custom.min()}, max={X_custom.max()}")
    
    # Ensure data has the correct channel dimension
    print(f"X_custom shape before checking channel dimension: {X_custom.shape}")
    if len(X_custom.shape) == 3:
        print(f"Adding channel dimension to custom images")
        X_custom = X_custom.reshape(X_custom.shape + (1,))
    
    # Make sure we have exactly 1 channel (grayscale) for model input
    if X_custom.shape[3] != 1:
        print(f"Unexpected channel count: {X_custom.shape[3]}, reshaping to single channel")
        if X_custom.shape[3] == 3:
            # Convert RGB to grayscale
            print("Converting RGB to grayscale")
            grayscale = 0.299 * X_custom[:, :, :, 0] + 0.587 * X_custom[:, :, :, 1] + 0.114 * X_custom[:, :, :, 2]
            X_custom = grayscale.reshape(grayscale.shape + (1,))
        else:
            # Just take the first channel
            X_custom = X_custom[:, :, :, 0:1]
    
    print(f"X_custom final shape: {X_custom.shape}")
    print(f"X_custom data range: [{X_custom.min():.6f}, {X_custom.max():.6f}]")
    
    # Create y_custom for actual class labels
    y_custom = np.array(range(len(image_paths)))  # Fix missing y_custom definition
    
    # Get predictions
    predictions = get_top_k_predictions(params, X_custom)
    
    for i in range(len(image_paths)):
        print(f"Prediction for image {i+1}:")
        plot_image_statistics(predictions, i, image_paths[i], sign_names, X_custom)
        print("---------------------------------------------------------------------------------------------------\n")
    
    # Filter valid classes (< 99)
    valid_indices = y_custom < 99
    X_custom_valid = X_custom[valid_indices]
    y_custom_valid = y_custom[valid_indices]
    
    # Only calculate accuracy if we have valid classes
    if len(X_custom_valid) > 0:
        y_custom_onehot = tf.one_hot(y_custom_valid, 43).numpy()  # Using TF 2.x one_hot
        predictions_valid = get_top_k_predictions(params, X_custom_valid)[1][:, 0]
        accuracy = 100.0 * np.sum(predictions_valid == np.argmax(y_custom_onehot, 1)) / predictions_valid.shape[0]
        print(f"Accuracy on captured images: {accuracy:.2f}%")

def plot_image_statistics(predictions, index, image_path, sign_names, X_custom):
    """
    Plot original image, preprocessed image, and prediction probabilities
    Args:
        predictions: model predictions
        index: image index
        image_path: path to image
        sign_names: dictionary of sign names
        X_custom: preprocessed images
    """
    original = io.imread(image_path)
    
    # Extract the preprocessed image correctly - X_custom has shape (samples, height, width, channels)
    if X_custom.shape[3] == 1:
        # Get the single channel for display
        preprocessed = X_custom[index, :, :, 0]
    else:
        # Just in case there are multiple channels
        preprocessed = X_custom[index]
        if len(preprocessed.shape) > 2:
            # Take first channel for display
            preprocessed = preprocessed[:, :, 0]
    
    plt.figure(figsize=(6, 2))
    plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
    plt.imshow(original)
    plt.axis('off')
    plt.title("Original Image")
    
    plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    plt.title("Preprocessed Image")
    
    plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=2)
    plt.barh(np.arange(5)+.5, predictions[0][index], align='center')
    plt.yticks(np.arange(5)+.5, [sign_names[i] for i in predictions[1][index].astype(int)])
    plt.title("Top 5 Predictions")
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.show()

def get_top_k_predictions(params, X, k=5):
    """
    Get top-k predictions for images
    Args:
        params: model parameters
        X: preprocessed images
        k: number of top predictions to return
    Returns:
        numpy array of top-k predictions
    """
    paths = Paths(params)
    
    # Debug point: Model loading
    print(f"Loading model from: {paths.model_path}")
    
    try:
        # First attempt: Try loading as a full Keras model
        model = tf.keras.models.load_model(paths.root_path)
        print("Loaded full Keras model successfully")
    except Exception as e:
        print(f"Could not load full Keras model: {e}")
        model = None
        
    if model is None:
        try:
            # Second attempt: Build Keras model and load weights
            print("Building Keras model and loading weights...")
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(params.image_size[0], params.image_size[1], 1)))
            
            # Layer 1: Convolutional + MaxPooling
            model.add(tf.keras.layers.Conv2D(params.conv1_d, (params.conv1_k, params.conv1_k), 
                                             activation='relu', padding='same', name='conv1'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
            model.add(tf.keras.layers.Dropout(1 - params.conv1_p, name='dropout1'))
            
            # Layer 2: Convolutional + MaxPooling
            model.add(tf.keras.layers.Conv2D(params.conv2_d, (params.conv2_k, params.conv2_k), 
                                             activation='relu', padding='same', name='conv2'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
            model.add(tf.keras.layers.Dropout(1 - params.conv2_p, name='dropout2'))
            
            # Layer 3: Convolutional + MaxPooling
            model.add(tf.keras.layers.Conv2D(params.conv3_d, (params.conv3_k, params.conv3_k), 
                                             activation='relu', padding='same', name='conv3'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3'))
            model.add(tf.keras.layers.Dropout(1 - params.conv3_p, name='dropout3'))
            
            # Flatten and fully connected layers
            model.add(tf.keras.layers.Flatten(name='flatten'))
            model.add(tf.keras.layers.Dense(params.fc4_size, activation='relu', 
                                            kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) 
                                            if params.l2_reg_enabled else None, name='fc4'))
            model.add(tf.keras.layers.Dropout(1 - params.fc4_p, name='dropout4'))
            model.add(tf.keras.layers.Dense(params.num_classes, activation='softmax', name='output'))
            
            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Load weights
            model.load_weights(paths.model_path)
            print("Loaded model weights successfully")
        except Exception as e:
            print(f"Could not load model weights: {e}")
            
            # Final fallback: Use legacy API with model_pass
            print("Falling back to TF compat mode with model_pass function...")
            
            # Create a function to run in graph mode that doesn't rely on is_training being a tensor
            def run_model_in_graph_mode(X):
                # Use tf.compat.v1 for compatibility with older saved models
                # If eager execution is already disabled, this is a no-op
                try:
                    tf.compat.v1.disable_eager_execution()
                except:
                    print("Eager execution already disabled")
                
                graph = tf.compat.v1.Graph()
                with graph.as_default():
                    tf_x = tf.compat.v1.placeholder(tf.float32, shape=(None, params.image_size[0], params.image_size[1], 1))
                    # Use constant False instead of tensor
                    with tf.compat.v1.variable_scope(paths.var_scope):
                        # Call model_pass with explicit False for is_training
                        logits = model_pass(tf_x, params, False)
                        predictions = tf.nn.softmax(logits)
                        top_k_predictions = tf.nn.top_k(predictions, k)
                    
                    with tf.compat.v1.Session(graph=graph) as session:
                        session.run(tf.compat.v1.global_variables_initializer())
                        saver = tf.compat.v1.train.Saver()
                        try:
                            saver.restore(session, paths.model_path)
                            print("Model restored from checkpoint successfully")
                            [p] = session.run([top_k_predictions], feed_dict={tf_x: X})
                            return np.array(p)
                        except Exception as e:
                            print(f"Error loading checkpoint: {e}")
                            return None
            
            # Try the graph mode execution
            result = run_model_in_graph_mode(X)
            if result is not None:
                return result
            else:
                print("All loading approaches failed")
                return np.zeros((2, k, X.shape[0])) # Return empty predictions
    # Use Keras model for predictions
    if model is not None:
        # Make predictions
        print("Making predictions with Keras model...")
        predictions = model.predict(X)
        
        # Get top-k values and indices
        values, indices = tf.nn.top_k(predictions, k=k)
        return np.array([values.numpy(), indices.numpy()])
    else:
        print("No model was successfully loaded")
        return np.zeros((2, k, X.shape[0])) # Return empty predictions

class Paths:
    """Class to handle model paths and naming"""
    def __init__(self, params):
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        
        # Debug point: Path initialization
        print(f"Model name: {self.model_name}")
        print(f"Model path: {self.model_path}")
        
        # Check if model dir exists, if not try alternatives
        if not os.path.exists(self.root_path):
            # Try the opposite learning rate decay setting
            alt_model_name = self.get_model_name(params, flip_lrdec=True)
            alt_root_path = os.getcwd() + "/models/" + alt_model_name + "/"
            
            if os.path.exists(alt_root_path):
                print(f"Model directory not found at {self.root_path}")
                print(f"Using alternative model path: {alt_root_path}")
                self.model_name = alt_model_name
                self.root_path = alt_root_path
                self.model_path = self.get_model_path()
    
    def get_model_name(self, params, flip_lrdec=False):
        model_name = f"k{params.conv1_k}d{params.conv1_d}p{params.conv1_p}_k{params.conv2_k}d{params.conv2_d}p{params.conv2_p}_k{params.conv3_k}d{params.conv3_d}p{params.conv3_p}_fc{params.fc4_size}p{params.fc4_p}"
        
        # Apply learning rate decay setting
        use_lrdec = params.learning_rate_decay
        if flip_lrdec:
            use_lrdec = not use_lrdec
        
        model_name += "_lrdec" if use_lrdec else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name
    
    def get_variables_scope(self, params):
        var_scope = f"k{params.conv1_k}d{params.conv1_d}_k{params.conv2_k}d{params.conv2_d}_k{params.conv3_k}d{params.conv3_d}_fc{params.fc4_size}_fc0"
        return var_scope
    
    def get_model_path(self):
        # Check for different possible model file paths
        possible_paths = [
            self.root_path + "model.weights.h5",  # Keras weights file
            self.root_path + "model.keras",       # Full model in Keras 3 format
            self.root_path + "model.h5",          # Full model in legacy H5 format
            self.root_path + "model.ckpt",        # TensorFlow checkpoint
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Default to weights path even if it doesn't exist
        return self.root_path + "model.weights.h5"