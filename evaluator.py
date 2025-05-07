# evaluator.py
import os
import tensorflow as tf
import numpy as np
from model_builder import Parameters, model_pass

def get_top_k_predictions(params, X, k=5):
    """
    Get the top k predictions for input images X using the model defined by params.
    
    Args:
        params: Parameters object containing model configuration
        X: Input images array of shape (batch_size, height, width, 1)
        k: Number of top predictions to return
        
    Returns:
        Numpy array containing top k values and indices
    """
    # Initialize paths for the model
    paths = Paths(params)
    
    # Create a new TensorFlow graph
    graph = tf.Graph()
    with graph.as_default():
        # Debug: Print graph being created
        print(f"Creating graph with model scope: {paths.var_scope}")
        
        # Define input placeholder
        tf_x = tf.placeholder(tf.float32, shape=(None, params.image_size[0], params.image_size[1], 1))
        
        # Set training mode to False for inference
        is_training = tf.constant(False)
        
        # Build the model within the specified variable scope
        with tf.variable_scope(paths.var_scope):
            # Debug: Print model architecture parameters
            print(f"Building model with conv layers: {params.conv1_d}, {params.conv2_d}, {params.conv3_d}")
            print(f"FC layer size: {params.fc4_size}")
            
            # Get logits from model_pass and apply softmax
            logits = model_pass(tf_x, params, is_training)
            predictions = tf.nn.softmax(logits)
            
            # Debug: Print tensor shapes
            print(f"Input shape: {tf_x.shape}, Predictions shape: {predictions.shape}")
        
        # Get top k predictions (values and indices)
        top_k_predictions = tf.nn.top_k(predictions, k)
        
        # Create a session and restore model weights
        with tf.Session(graph=graph) as session:
            # Debug: Print model path
            print(f"Loading model from: {paths.model_path}")
            
            # Initialize variables
            session.run(tf.global_variables_initializer())
            
            try:
                # Restore model weights
                tf.train.Saver().restore(session, paths.model_path)
                print("Model restored successfully!")
            except Exception as e:
                print(f"Error restoring model: {e}")
                raise
            
            # Debug: Print input batch info
            print(f"Running inference on batch of {X.shape[0]} images")
            
            try:
                # Run inference
                [p] = session.run([top_k_predictions], feed_dict={tf_x: X})
                print(f"Inference successful, prediction shape: {p.values.shape}")
                return np.array(p)
            except Exception as e:
                print(f"Error during inference: {e}")
                raise


class Paths:
    """
    Helper class to manage model paths and naming based on model parameters.
    """
    def __init__(self, params):
        """Initialize paths based on model parameters"""
        # Debug: Print parameters being used
        print(f"Initializing paths with image size: {params.image_size}")
        
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        
        # Root path for model storage
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        
        # Debug: Check if directory exists
        if not os.path.exists(self.root_path):
            print(f"Warning: Model directory does not exist: {self.root_path}")
        
        self.model_path = self.get_model_path()
        
        # Debug: Print final paths
        print(f"Model name: {self.model_name}")
        print(f"Variable scope: {self.var_scope}")
        print(f"Model path: {self.model_path}")
        
    def get_model_name(self, params):
        """Generate a model name based on its architecture parameters"""
        # Build model name from convolutional layer parameters
        model_name = f"k{params.conv1_k}d{params.conv1_d}p{params.conv1_p}_"
        model_name += f"k{params.conv2_k}d{params.conv2_d}p{params.conv2_p}_"
        model_name += f"k{params.conv3_k}d{params.conv3_d}p{params.conv3_p}_"
        model_name += f"fc{params.fc4_size}p{params.fc4_p}"
        
        # Add learning rate decay and L2 regularization info
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        
        return model_name
    
    def get_variables_scope(self, params):
        """Generate the variable scope name based on model architecture"""
        # Create variable scope name (simpler than model_name)
        var_scope = f"k{params.conv1_k}d{params.conv1_d}_"
        var_scope += f"k{params.conv2_k}d{params.conv2_d}_"
        var_scope += f"k{params.conv3_k}d{params.conv3_d}_"
        var_scope += f"fc{params.fc4_size}_fc0"
        
        return var_scope
    
    def get_model_path(self):
        """Get the full path to the model checkpoint"""
        return self.root_path + "model.ckpt"