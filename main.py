# main.py
import pickle
import argparse
import json
import numpy as np
from data_loader import load_pickled_data, load_sign_names
from data_processor import preprocess_dataset, flip_extend, extend_balancing_classes
from trainer import train_model, Parameters
from sklearn.model_selection import train_test_split
import os
import glob
from skimage import io
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Classification')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], default='train')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--create-visualization', action='store_true', help='Create extended dataset for class distribution visualization')
    args = parser.parse_args()
    
    print("Loading configuration...")
    with open(args.config) as f:
        config = json.load(f)
    
    # Check if pickle files exist and have content
    train_pickle_path = 'data/train.p'
    test_pickle_path = 'data/test.p'
    
    # Create train.p from raw images if it doesn't exist or is empty
    if not os.path.exists(train_pickle_path) or os.path.getsize(train_pickle_path) == 0:
        print("Train pickle file is empty or doesn't exist. Creating from raw images...")
        train_data = create_train_pickle_from_raw()
        with open(train_pickle_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"Created {train_pickle_path} with {len(train_data['features'])} images")
    
    # Create test.p from raw images if it doesn't exist or is empty
    if not os.path.exists(test_pickle_path) or os.path.getsize(test_pickle_path) == 0:
        print("Test pickle file is empty or doesn't exist. Creating from raw images...")
        test_data = create_test_pickle_from_raw()
        with open(test_pickle_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"Created {test_pickle_path} with {len(test_data['features'])} images")
    
    print("Loading training and test datasets...")
    X_train, y_train = load_pickled_data('data/train.p', ['features', 'labels'])
    X_test, y_test = load_pickled_data('data/test.p', ['features', 'labels'])
    
    print("Preprocessing training and test datasets...")
    X_train, y_train = preprocess_dataset(X_train, y_train)
    X_test, y_test = preprocess_dataset(X_test, y_test)
    
    # Only create balanced dataset if needed and doesn't exist already
    balanced_pickle_path = 'data/train_balanced.p'
    if args.mode == 'train' and (not os.path.exists(balanced_pickle_path) or os.path.getsize(balanced_pickle_path) == 0):
        print("Creating balanced dataset...")
        X_train_balanced, y_train_balanced = extend_balancing_classes(X_train, y_train, aug_intensity=0.75, counts=np.full(43, 20000, dtype=int))
        
        print("Saving balanced dataset to disk...")
        with open(balanced_pickle_path, 'wb') as f:
            pickle.dump({'features': X_train_balanced, 'labels': y_train_balanced}, f)
    
    # Only create extended dataset for visualization if explicitly requested
    if args.create_visualization:
        extended_pickle_path = 'data/train_extended.p'
        if not os.path.exists(extended_pickle_path) or os.path.getsize(extended_pickle_path) == 0:
            print("Creating extended dataset for class distribution visualization...")
            X_train_extended, y_train_extended = extend_balancing_classes(X_train, y_train, aug_intensity=0.75, counts=np.array([np.sum(y_train == c) for c in range(43)]) * 20)
            
            print("Saving extended dataset to disk...")
            with open(extended_pickle_path, 'wb') as f:
                pickle.dump({'features': X_train_extended, 'labels': y_train_extended}, f)
    
    if args.mode == 'train':
        print("Training mode selected.")
        print("Loading balanced dataset from disk...")
        X_train_balanced, y_train_balanced = load_pickled_data('data/train_balanced.p', ['features', 'labels'])
        print("Splitting data into training and validation sets...")
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_balanced, y_train_balanced, test_size=0.25)
        
        print("Setting model parameters...")
        parameters = Parameters(
            num_classes=43,
            image_size=(32, 32),
            batch_size=256,
            max_epochs=400,
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
        
        print("Starting model training...")
        train_model(parameters, X_train, y_train, X_valid, y_valid, X_test, y_test, config["logger_config"])
    
    elif args.mode == 'evaluate':
        print("Evaluation mode selected.")
        from evaluator import get_top_k_predictions
        print("Setting model parameters for evaluation...")
        parameters = Parameters(
            num_classes=43,
            image_size=(32, 32),
            batch_size=256,
            max_epochs=400,
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
        print("Generating predictions on test dataset...")
        predictions = get_top_k_predictions(parameters, X_test)
        predicted_labels = predictions[1][:, np.argmax(predictions[0], 1)][:, 0].astype(int)
        true_labels = np.argmax(y_test, 1)
        accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    elif args.mode == 'predict':
        print("Prediction mode selected.")
        from predictor import predict_on_new_images
        print("Setting model parameters for prediction...")
        parameters = Parameters(
            num_classes=43,
            image_size=(32, 32),
            batch_size=256,
            max_epochs=400,
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
        
        # Create custom folder if it doesn't exist
        os.makedirs('data/custom', exist_ok=True)
        
        print("Collecting custom images for prediction...")
        custom_image_paths = [os.path.join('data/custom', f) for f in os.listdir('data/custom') if f.endswith('.png')]
        if not custom_image_paths:
            print("No custom images found. Copying some test images as examples...")
            # Copy some test images to custom folder for prediction
            import shutil
            test_images = glob.glob('data/test/Images/*.ppm')[:5]  # Get first 5 test images
            for i, img_path in enumerate(test_images):
                output_path = f'data/custom/test_example_{i}.png'
                # Convert PPM to PNG
                img = io.imread(img_path)
                io.imsave(output_path, img)
                custom_image_paths.append(output_path)
            
        print("Starting prediction on custom images...")
        predict_on_new_images(parameters, custom_image_paths)

def create_train_pickle_from_raw():
    """Create a pickle file from raw train images"""
    print("Loading training images from data/train folder...")
    from skimage import transform
    
    features = []
    labels = []
    target_size = (32, 32, 3)  # Standard size for traffic signs
    
    # Each subfolder in train directory is a class
    class_folders = sorted(glob.glob('data/train/*/'))
    
    for class_idx, class_folder in enumerate(class_folders):
        class_images = glob.glob(os.path.join(class_folder, '*.ppm'))
        print(f"Processing class {class_idx}: {len(class_images)} images")
        
        for img_path in class_images:
            try:
                img = io.imread(img_path)
                # Resize image to target size
                if img.shape != target_size:
                    img = transform.resize(img, (32, 32), anti_aliasing=True)
                    img = (img * 255).astype(np.uint8)
                # Ensure image has 3 channels
                if len(img.shape) == 2:  # Grayscale image
                    img = np.stack((img,) * 3, axis=-1)
                features.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Loaded {len(features)} training images with shape {features.shape}")
    
    return {
        'features': features,
        'labels': labels
    }

def create_test_pickle_from_raw():
    """Create a pickle file from raw test images"""
    print("Loading test images from data/test/Images folder...")
    from skimage import transform
    
    features = []
    labels = []
    target_size = (32, 32, 3)  # Standard size for traffic signs
    
    # Extract class name from filename
    test_images = sorted(glob.glob('data/test/Images/*.ppm'))
    
    for img_path in test_images:
        try:
            # Just use a dummy label (0) for test images since we'll need manual annotation anyway
            # We can update this later if we get the actual labels
            img_class = 0
            
            img = io.imread(img_path)
            # Resize image to target size
            if img.shape != target_size:
                img = transform.resize(img, (32, 32), anti_aliasing=True)
                img = (img * 255).astype(np.uint8)
            # Ensure image has 3 channels
            if len(img.shape) == 2:  # Grayscale image
                img = np.stack((img,) * 3, axis=-1)
            
            features.append(img)
            labels.append(img_class)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Loaded {len(features)} test images with shape {features.shape}")
    
    return {
        'features': features,
        'labels': labels
    }

if __name__ == '__main__':
    main()
