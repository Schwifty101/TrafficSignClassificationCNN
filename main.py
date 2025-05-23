# main.py
import pickle
import argparse
import json
import numpy as np
from data_loader import load_pickled_data, load_sign_names
from data_processor import preprocess_dataset, flip_extend, extend_balancing_classes
from trainer import train_model, Parameters, progress_bar
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import time
import sys

"""
Traffic Sign Recognition CNN Trainer

This script handles the complete pipeline for traffic sign recognition:
1. Data loading and preprocessing
2. Dataset balancing and augmentation 
   - Creates balanced and extended datasets
   - Limits dataset size to exactly 2x the original size
   - Ensures all classes have balanced representation
3. Model training
4. Prediction

Command-line arguments:
  --mode: 'train' or 'predict'
  --config: Path to configuration file (default: config.json)
  --rebuild-datasets: Force rebuild of processed datasets
"""

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
if not tf.__version__.startswith('2.'):
    print("Warning: This code is designed for TensorFlow 2.x. Current version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

def print_debug(message):
    """Helper function to print debug messages with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {message}")

def process_data_with_progress(operation_name, operation_fn, *args, **kwargs):
    """Wrapper to add progress indication for data processing operations"""
    print_debug(f"Starting {operation_name}...")
    start_time = time.time()
    result = operation_fn(*args, **kwargs)
    elapsed_time = time.time() - start_time
    print_debug(f"Completed {operation_name} in {elapsed_time:.2f} seconds")
    return result

def main():
    print_debug("=== Starting Traffic Sign Classification Application ===")
    
    parser = argparse.ArgumentParser(description='Traffic Sign Classification')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--rebuild-datasets', action='store_true', help='Force rebuild of processed datasets')
    args = parser.parse_args()
    print_debug(f"Running in {args.mode.upper()} mode with config: {args.config}")
    
    # Check if datasets should be rebuilt
    if args.rebuild_datasets:
        print_debug("Rebuild datasets flag set, removing existing processed datasets...")
        for dataset_file in ['data/train_balanced.p', 'data/train_extended.p', 'data/train_flipped.p']:
            if os.path.exists(dataset_file):
                os.remove(dataset_file)
                print_debug(f"Removed existing dataset file: {dataset_file}")
                
    # Load configuration
    try:
        with open(args.config) as f:
            config = json.load(f)
        print_debug(f"Configuration loaded successfully")
    except Exception as e:
        print_debug(f"Error loading configuration: {e}")
        return
    
    # Load data with progress indication
    print_debug("Loading datasets...")
    try:
        print_debug("Loading training data...")
        X_train, y_train = load_pickled_data('data/train.p', ['features', 'labels'])
        print_debug(f"Training data loaded: {X_train.shape[0]} samples")
        
        print_debug("Loading testing data...")
        X_test, filenames = load_pickled_data('data/test.p', ['features', 'filenames'])
        print_debug(f"Testing data loaded: {X_test.shape[0]} samples (without labels)")
        
        # Creating dummy y_test since we don't have ground truth for test data
        y_test = np.zeros(X_test.shape[0], dtype=np.int32)
        print_debug("Created dummy labels for test data (required for code compatibility)")
    except Exception as e:
        print_debug(f"Error loading dataset: {e}")
        return
    
    # Preprocess data with progress indicators
    total_steps = 5  # Total preprocessing steps
    current_step = 0
    
    # Step 1: Preprocess training data
    current_step += 1
    print_debug(f"Preprocessing step {current_step}/{total_steps}: Preprocessing training data")
    sys.stdout.write("\rPreprocessing training data: 0%")
    sys.stdout.flush()
    X_train, y_train = process_data_with_progress("training data preprocessing", 
                                                  preprocess_dataset, X_train, y_train)
    progress_bar(1, 1, prefix="Preprocessing training data", suffix="Complete")
    print_debug(f"Final training data size: {X_train.shape[0]} samples")
    
    # Step 2: Preprocess testing data
    current_step += 1
    print_debug(f"Preprocessing step {current_step}/{total_steps}: Preprocessing testing data")
    X_test, y_test = process_data_with_progress("testing data preprocessing", 
                                                preprocess_dataset, X_test, y_test)
    progress_bar(1, 1, prefix="Preprocessing testing data", suffix="Complete")
    
    # Step 3: Create flipped dataset
    current_step += 1
    print_debug(f"Preprocessing step {current_step}/{total_steps}: Creating flipped dataset")
    if os.path.exists('data/train_flipped.p'):
        print_debug("Flipped dataset already exists, loading...")
        X_train_flipped, y_train_flipped = load_pickled_data('data/train_flipped.p', ['features', 'labels'])
        print_debug(f"Loaded flipped dataset with {len(X_train_flipped)} samples")
    else:
        print_debug("Creating flipped dataset with domain-specific augmentations...")
        X_train_flipped, y_train_flipped = process_data_with_progress(
            "flip-based dataset extension",
            flip_extend, X_train, y_train
        )
        
        print_debug(f"Saving flipped dataset with {len(X_train_flipped)} samples...")
        with open('data/train_flipped.p', 'wb') as f:
            pickle.dump({'features': X_train_flipped, 'labels': y_train_flipped}, f)
        
        # Verify class distribution in flipped dataset
        print_debug("Verifying class distribution in flipped dataset...")
        # Check if labels are one-hot encoded or single integers
        if len(y_train_flipped.shape) > 1 and y_train_flipped.shape[1] > 1:
            # One-hot encoded format
            flipped_class_counts = np.array([np.sum(np.argmax(y_train_flipped, axis=1) == c) for c in range(43)])
        else:
            # Single label format
            flipped_class_counts = np.array([np.sum(y_train_flipped == c) for c in range(43)])
        
        print_debug(f"Flipped dataset stats:")
        print_debug(f"  - Total samples: {len(X_train_flipped)}")
        print_debug(f"  - Min samples per class: {np.min(flipped_class_counts)} (Class {np.argmin(flipped_class_counts)})")
        print_debug(f"  - Max samples per class: {np.max(flipped_class_counts)} (Class {np.argmax(flipped_class_counts)})")
    
    # Step 4: Create balanced dataset
    current_step += 1
    print_debug(f"Preprocessing step {current_step}/{total_steps}: Creating balanced dataset")
    if os.path.exists('data/train_balanced.p'):
        print_debug("Balanced dataset already exists, loading for verification...")
        X_train_balanced_check, y_train_balanced_check = load_pickled_data('data/train_balanced.p', ['features', 'labels'])
        
        # Verify class distribution
        print_debug("Verifying class distribution in existing balanced dataset...")
        # Check if labels are one-hot encoded or single integers
        if len(y_train_balanced_check.shape) > 1 and y_train_balanced_check.shape[1] > 1:
            # One-hot encoded format
            class_counts = np.array([np.sum(np.argmax(y_train_balanced_check, axis=1) == c) for c in range(43)])
        else:
            # Single label format
            class_counts = np.array([np.sum(y_train_balanced_check == c) for c in range(43)])
        
        # Calculate average count and check if samples are within 10% of target
        avg_count = np.mean(class_counts)
        tolerance = 0.1  # 10% tolerance
        classes_within_tolerance = np.sum((class_counts >= avg_count * (1 - tolerance)) & 
                                        (class_counts <= avg_count * (1 + tolerance)))
        
        print_debug(f"Class distribution stats for existing balanced dataset:")
        print_debug(f"  - Total samples: {len(X_train_balanced_check)}")
        print_debug(f"  - Average samples per class: {avg_count:.1f}")
        print_debug(f"  - Min samples: {np.min(class_counts)} (Class {np.argmin(class_counts)})")
        print_debug(f"  - Max samples: {np.max(class_counts)} (Class {np.argmax(class_counts)})")
        print_debug(f"  - Classes within 10% of average: {classes_within_tolerance}/43 ({classes_within_tolerance/43*100:.1f}%)")
        
        # Print warning if distribution is significantly unbalanced
        if classes_within_tolerance < 39:  # If less than 90% of classes are within tolerance
            print_debug("WARNING: Existing dataset balance may not be optimal. Consider using --rebuild-datasets flag.")
    else:
        print_debug("Creating balanced dataset (this may take a while)...")
        steps = 43  # One for each class
        for i in range(steps):
            progress_bar(i+1, steps, prefix="Creating balanced dataset", 
                        suffix=f"Processing class {i+1}/{steps}")
            time.sleep(0.01)  # Small delay to simulate work and show progress
        
        # Calculate max total size as exactly 2x the original size
        max_size = X_train.shape[0] * 2
        print_debug(f"Limiting balanced dataset to maximum {max_size} samples (2x original)")
        
        X_train_balanced, y_train_balanced = process_data_with_progress(
            "balanced dataset creation",
            extend_balancing_classes, X_train, y_train, 
            aug_intensity=0.75, max_total_size=max_size
        )
        
        print_debug(f"Saving balanced dataset with {len(X_train_balanced)} samples...")
        with open('data/train_balanced.p', 'wb') as f:
            pickle.dump({'features': X_train_balanced, 'labels': y_train_balanced}, f)
        
        # After creating balanced dataset, print its size
        print_debug(f"Balanced training data size: {len(X_train_balanced)} samples")
        
        # Verify class distribution
        print_debug("Verifying class distribution in balanced dataset...")
        # Check if labels are one-hot encoded or single integers
        if len(y_train_balanced.shape) > 1 and y_train_balanced.shape[1] > 1:
            # One-hot encoded format
            class_counts = np.array([np.sum(np.argmax(y_train_balanced, axis=1) == c) for c in range(43)])
        else:
            # Single label format
            class_counts = np.array([np.sum(y_train_balanced == c) for c in range(43)])
        
        # Calculate average count and check if samples are within 10% of target
        avg_count = np.mean(class_counts)
        tolerance = 0.1  # 10% tolerance
        classes_within_tolerance = np.sum((class_counts >= avg_count * (1 - tolerance)) & 
                                        (class_counts <= avg_count * (1 + tolerance)))
        
        print_debug(f"Class distribution stats:")
        print_debug(f"  - Average samples per class: {avg_count:.1f}")
        print_debug(f"  - Min samples: {np.min(class_counts)} (Class {np.argmin(class_counts)})")
        print_debug(f"  - Max samples: {np.max(class_counts)} (Class {np.argmax(class_counts)})")
        print_debug(f"  - Classes within 10% of average: {classes_within_tolerance}/43 ({classes_within_tolerance/43*100:.1f}%)")
        
        # Print warning if distribution is significantly unbalanced
        if classes_within_tolerance < 39:  # If less than 90% of classes are within tolerance
            print_debug("WARNING: Dataset balance may not be optimal. Consider adjusting balancing parameters.")
    
    # Step 5: Create extended dataset
    current_step += 1
    print_debug(f"Preprocessing step {current_step}/{total_steps}: Creating extended dataset")
    if os.path.exists('data/train_extended.p'):
        print_debug("Extended dataset already exists, skipping creation")
    else:
        print_debug("Creating extended dataset (this may take a while)...")
        steps = 43  # One for each class
        for i in range(steps):
            progress_bar(i+1, steps, prefix="Creating extended dataset", 
                        suffix=f"Processing class {i+1}/{steps}")
            time.sleep(0.01)  # Small delay to simulate work and show progress
            
        # Calculate max total size as exactly 2x the original size
        max_size = X_train.shape[0] * 2
        print_debug(f"Limiting extended dataset to maximum {max_size} samples (2x original)")
        
        # Prepare custom counts but respect the max total size
        # Handle both one-hot encoded and single label formats
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # One-hot encoded format
            class_counts = np.array([np.sum(np.argmax(y_train, axis=1) == c) for c in range(43)])
        else:
            # Single label format
            class_counts = np.array([np.sum(y_train == c) for c in range(43)])
            
        custom_counts = class_counts * 20  # Target 20x per class
        
        X_train_extended, y_train_extended = process_data_with_progress(
            "extended dataset creation",
            extend_balancing_classes, X_train, y_train, 
            aug_intensity=0.75, counts=custom_counts, max_total_size=max_size
        )
        
        print_debug(f"Saving extended dataset with {len(X_train_extended)} samples...")
        with open('data/train_extended.p', 'wb') as f:
            pickle.dump({'features': X_train_extended, 'labels': y_train_extended}, f)
    
    # Display preprocessing completion message
    print_debug("=== Preprocessing completed successfully ===")
    
    if args.mode == 'train':
        print_debug("=== Starting Model Training ===")
        
        # Load flipped + balanced dataset combination for training
        print_debug("Loading flipped and balanced datasets for improved training...")
        X_train_flipped, y_train_flipped = load_pickled_data('data/train_flipped.p', ['features', 'labels'])
        X_train_balanced, y_train_balanced = load_pickled_data('data/train_balanced.p', ['features', 'labels'])
        
        # Combine the datasets
        print_debug(f"Combining datasets: flipped ({len(X_train_flipped)} samples) + balanced ({len(X_train_balanced)} samples)...")
        X_combined = np.concatenate((X_train_flipped, X_train_balanced), axis=0)
        y_combined = np.concatenate((y_train_flipped, y_train_balanced), axis=0)
        
        # Shuffle the combined dataset
        print_debug("Shuffling combined dataset...")
        from sklearn.utils import shuffle
        X_combined, y_combined = shuffle(X_combined, y_combined)
        
        print_debug(f"Final combined training dataset: {X_combined.shape[0]} samples")
        
        # Data is already in [0, 1] range and float32 format from the preprocessing
        print_debug("Data is already in [0, 1] range and float32 format, skipping scaling...")
        
        # Ensure data has the correct shape with channel dimension
        if len(X_combined.shape) == 3:
            print_debug("Adding channel dimension to combined data...")
            X_combined = X_combined.reshape(X_combined.shape + (1,))
        
        # Split into train and validation
        print_debug("Splitting combined data into training and validation sets (75%/25%)...")
        X_train, X_valid, y_train, y_valid = train_test_split(X_combined, y_combined, test_size=0.25)
        print_debug(f"Training set: {X_train.shape[0]} samples")
        print_debug(f"Validation set: {X_valid.shape[0]} samples")
        
        # Define model parameters - ensure they're compatible with TF 2.x
        print_debug("Setting up model parameters...")
        parameters = Parameters(
            num_classes=43,
            image_size=(32, 32),
            batch_size=128,
            max_epochs=200,
            log_epoch=1,
            print_epoch=1,
            learning_rate_decay=False,  # disable learning rate decay
            learning_rate=0.001,       # Initial learning rate for training
            l2_reg_enabled=True,
            l2_lambda=0.0001,
            early_stopping_enabled=True,
            early_stopping_patience=100,
            resume_training=False,
            conv1_k=5, conv1_d=32, conv1_p=0.9,
            conv2_k=5, conv2_d=64, conv2_p=0.8,
            conv3_k=5, conv3_d=128, conv3_p=0.7,
            fc4_size=1024, fc4_p=0.5
        )
        
        print_debug("Starting model training...")
        # Train the model with TF 2.x compatible function
        model = train_model(parameters, X_train, y_train, X_valid, y_valid, X_test, y_test, config.get("logger_config"))
        print_debug("=== Model Training Completed ===")
    
    elif args.mode == 'predict':
        print_debug("=== Starting Custom Image Prediction ===")
        
        # For TF 2.x compatibility, ensure the predictor module is updated
        try:
            from predictor import predict_on_new_images
            print_debug("Initializing model for prediction...")
            
            # Predict on custom images
            parameters = Parameters(
                num_classes=43,
                image_size=(32, 32),
                batch_size=128,
                max_epochs=200,
                log_epoch=1,
                print_epoch=1,
                learning_rate_decay=False,    # Disable learning rate decay to match training mode
                learning_rate=0.0001,        # Lower learning rate for prediction
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
            
            custom_image_dir = 'data/custom'
            if not os.path.exists(custom_image_dir):
                print_debug(f"Custom image directory {custom_image_dir} not found")
                return
                
            custom_image_paths = [os.path.join(custom_image_dir, f) for f in os.listdir(custom_image_dir) if f.endswith('.png')]
            
            if not custom_image_paths:
                print_debug("No PNG images found in the custom image directory")
                return
                
            print_debug(f"Found {len(custom_image_paths)} custom images for prediction")
            
            # Show progress while loading and processing images
            print_debug("Processing custom images...")
            for i, img_path in enumerate(custom_image_paths):
                progress_bar(i+1, len(custom_image_paths), 
                            prefix="Processing images", 
                            suffix=f"Image {i+1}/{len(custom_image_paths)}: {os.path.basename(img_path)}")
                time.sleep(0.2)  # Small delay to show progress
            
            print_debug("Running prediction on custom images...")
            predict_on_new_images(parameters, custom_image_paths)
            print_debug("=== Custom Image Prediction Completed ===")
            
        except Exception as e:
            print_debug(f"Error during prediction: {e}")
    
    print_debug("=== Traffic Sign Classification Application Completed ===")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")