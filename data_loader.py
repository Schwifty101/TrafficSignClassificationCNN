# data_loader.py
import pickle

def load_pickled_data(file, columns):
    """
    Loads pickled training and test data.
    """
    print(f"Loading pickled data from {file} with columns {columns}...")
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    print("Pickled data loaded successfully.")
    return tuple(map(lambda c: dataset[c], columns))

def load_sign_names(file='data/signnames.csv'):
    """
    Loads sign names from CSV file.
    """
    print(f"Loading sign names from {file}...")
    import pandas as pd
    sign_names = pd.read_csv(file).values[:, 1]
    print(f"{len(sign_names)} sign names loaded.")
    return sign_names
