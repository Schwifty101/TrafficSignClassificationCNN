# data_loader.py
import pickle
import os
import time

def load_pickled_data(file, columns):
    """
    Loads pickled training and test data.
    
    Args:
        file: Path to the pickle file
        columns: List of columns to extract from the pickle file
        
    Returns:
        Tuple of numpy arrays corresponding to the requested columns
    """
    # Debug: Print loading information
    print(f"DEBUG: Loading data from {file}")
    start_time = time.time()
    
    # Check if file exists
    if not os.path.exists(file):
        print(f"ERROR: File {file} not found!")
        raise FileNotFoundError(f"The file {file} does not exist")
    
    # Get file size for debugging
    file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
    print(f"DEBUG: File size: {file_size:.2f} MB")
    
    try:
        with open(file, mode='rb') as f:
            print(f"DEBUG: File opened successfully, loading data...")
            dataset = pickle.load(f)
            
        # Check if all requested columns exist in the dataset
        missing_columns = [col for col in columns if col not in dataset]
        if missing_columns:
            print(f"WARNING: The following columns were not found in the dataset: {missing_columns}")
            
        # Extract and return the requested columns
        result = tuple(map(lambda c: dataset[c], columns))
        
        # Print information about loaded data
        for i, col in enumerate(columns):
            if hasattr(result[i], 'shape'):
                print(f"DEBUG: Loaded column '{col}' with shape {result[i].shape}")
            else:
                print(f"DEBUG: Loaded column '{col}' (type: {type(result[i]).__name__})")
        
        elapsed_time = time.time() - start_time
        print(f"DEBUG: Data loading completed in {elapsed_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"ERROR: Failed to load data from {file}: {str(e)}")
        raise

def load_sign_names(file='data/signnames.csv'):
    """
    Loads sign names from CSV file.
    
    Args:
        file: Path to the CSV file containing sign names
        
    Returns:
        NumPy array of sign names
    """
    import pandas as pd
    
    # Debug: Print loading information
    print(f"DEBUG: Loading sign names from {file}")
    start_time = time.time()
    
    # Check if file exists
    if not os.path.exists(file):
        print(f"ERROR: Sign names file {file} not found!")
        raise FileNotFoundError(f"The file {file} does not exist")
    
    try:
        # Load the CSV file
        sign_names = pd.read_csv(file).values[:, 1]
        
        # Print information about loaded sign names
        print(f"DEBUG: Loaded {len(sign_names)} sign names")
        print(f"DEBUG: First few sign names: {sign_names[:5]}")
        
        elapsed_time = time.time() - start_time
        print(f"DEBUG: Sign names loading completed in {elapsed_time:.2f} seconds")
        
        return sign_names
        
    except Exception as e:
        print(f"ERROR: Failed to load sign names from {file}: {str(e)}")
        raise