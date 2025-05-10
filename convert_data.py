import os
import pickle
import numpy as np
from PIL import Image
import pandas as pd

def convert_train_to_pickle(train_dir, csv_file, output_file):
    X_train = []
    y_train = []
    
    # Load CSV for label mapping
    df = pd.read_csv(csv_file)
    
    # Iterate through subfolders (each subfolder name is a class ID)
    for folder in sorted(os.listdir(train_dir)):
        folder_path = os.path.join(train_dir, folder)
        if os.path.isdir(folder_path):
            # The folder name is the class ID (e.g., "00000" is class 0)
            try:
                label = int(folder)
                
                # Verify label exists in the signnames.csv
                if label not in df['ClassId'].values:
                    print(f"Warning: Class ID {label} not found in signnames.csv")
                    continue
                    
                # Process each image in the folder
                for img_file in os.listdir(folder_path):
                    if not img_file.endswith('.ppm'):
                        continue
                        
                    img_path = os.path.join(folder_path, img_file)
                    try:
                        img = Image.open(img_path).convert('RGB')  # Load as RGB
                        img = img.resize((32, 32))
                        img_array = np.array(img)
                        X_train.append(img_array)
                        y_train.append(label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            except ValueError:
                print(f"Skipping folder {folder} - not a class ID")
    
    # Convert lists to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Save to pickle
    with open(output_file, 'wb') as f:
        pickle.dump({'features': X_train, 'labels': y_train}, f)
    print(f"Saved {len(X_train)} training images to {output_file}")
    print(f"Data shape: {X_train.shape}, Labels shape: {y_train.shape}")

def convert_test_to_pickle(test_dir, output_file):
    X_test = []
    filenames = []
    
    # Process all test images
    for img_file in sorted(os.listdir(test_dir)):
        if not img_file.endswith('.ppm'):
            continue
            
        img_path = os.path.join(test_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((32, 32))
            img_array = np.array(img)
            X_test.append(img_array)
            filenames.append(img_file)  # Store filename for reference
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Convert lists to NumPy arrays
    X_test = np.array(X_test)
    
    # Save to pickle - no labels for test data
    with open(output_file, 'wb') as f:
        pickle.dump({
            'features': X_test,
            'filenames': filenames  # Store filenames instead of labels
        }, f)
    print(f"Saved {len(X_test)} test images to {output_file}")
    print(f"Data shape: {X_test.shape}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Train data
    train_dir = os.path.join(base_dir, 'data', 'train')
    csv_file = os.path.join(base_dir, 'data', 'signnames.csv')
    train_output = os.path.join(base_dir, 'data', 'train.p')
    
    # Test data
    test_dir = os.path.join(base_dir, 'data', 'test', 'Images')
    test_output = os.path.join(base_dir, 'data', 'test.p')
    
    print("Converting training data...")
    convert_train_to_pickle(train_dir, csv_file, train_output)
    
    print("\nConverting test data...")
    convert_test_to_pickle(test_dir, test_output)
    
    print("\nConversion complete!")
