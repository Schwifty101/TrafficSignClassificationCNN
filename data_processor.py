# data_processor.py
import numpy as np
from sklearn.utils import shuffle
from skimage import exposure
import warnings

def preprocess_dataset(X, y=None):
    """
    Performs feature scaling, one-hot encoding of labels and shuffles the data if labels are provided.
    """
    print(f"Preprocessing dataset with {X.shape[0]} examples:")
    
    # Check if X is already processed or has unexpected dimensions
    if len(X.shape) == 1:
        print(f"Warning: Input array has unexpected shape {X.shape}. Returning as is.")
        return X, y
    
    # Convert to grayscale if we have RGB data (4D array)
    if len(X.shape) == 4 and X.shape[3] == 3:
        X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    
    # Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)
    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i])
    
    if y is not None:
        # Convert to one-hot encoding
        num_classes = np.max(y) + 1
        y = np.eye(num_classes)[y]
        X, y = shuffle(X, y)
        # Add a single grayscale channel
        X = X.reshape(X.shape + (1,))
    return X, y

def flip_extend(X, y):
    """
    Extends existing images dataset by flipping images of some classes.
    """
    # Classes of signs that, when flipped horizontally, should still be classified as the same class
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Classes of signs that, when flipped vertically, should still be classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    # Classes of signs that, when flipped horizontally and then vertically, should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    # Classes of signs that, when flipped horizontally, would still be meaningful, but should be classified as some other class
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38],
    ])
    num_classes = 43
    
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y_indices.dtype)
    
    for c in range(num_classes):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y_indices == c], axis = 0)
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X[y_indices == c][:, :, ::-1, :], axis = 0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y_indices == flip_class][:, :, ::-1, :], axis = 0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        # If we can flip images of this class vertically and they would still belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis = 0)
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis = 0)
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    
    # Convert back to one-hot encoding if input was one-hot encoded
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_extended_one_hot = np.eye(num_classes)[y_extended.astype(int)]
        return (X_extended, y_extended_one_hot)
    else:
        return (X_extended, y_extended)

def extend_balancing_classes(X, y, aug_intensity=0.5, counts=None):
    """
    Extends dataset by duplicating existing images while applying data augmentation pipeline.
    """
    num_classes = 43
    
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
        
    _, class_counts = np.unique(y_indices, return_counts=True)
    max_c = max(class_counts)
    total = max_c * num_classes if counts is None else np.sum(counts)
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = np.float32)
    y_extended = np.empty([0], dtype = y_indices.dtype)
    
    print(f"Extending dataset using augmented data (intensity = {aug_intensity}):")
    
    # Progress tracking
    progress_total = num_classes
    progress_current = 0
    
    for c, c_count in zip(range(num_classes), class_counts):
        # Progress display
        progress_current += 1
        progress_percentage = progress_current / progress_total * 100
        progress_bar = '█' * int(progress_percentage / 2) + '-' * (50 - int(progress_percentage / 2))
        print(f"\r[{progress_bar}] {progress_percentage:.1f}% - Processing class {c}/{num_classes-1} ", end='', flush=True)
        
        # How many examples should there be eventually for this class:
        max_c = max_c if counts is None else counts[c]
        # First copy existing data for this class
        X_source = (X[y_indices == c] / 255.).astype(np.float32)
        y_source = y_indices[y_indices == c]
        X_extended = np.append(X_extended, X_source, axis = 0)
        
        for i in range((max_c // c_count) - 1):
            from utils.AugmentedSignsBatchIterator import AugmentedSignsBatchIterator
            batch_iterator = AugmentedSignsBatchIterator(batch_size = X_source.shape[0], p = 1.0, intensity = aug_intensity)
            for x_batch, _ in batch_iterator(X_source, y_source):
                X_extended = np.append(X_extended, x_batch, axis = 0)
        
        batch_iterator = AugmentedSignsBatchIterator(batch_size = max_c % c_count, p = 1.0, intensity = aug_intensity)
        for x_batch, _ in batch_iterator(X_source, y_source):
            X_extended = np.append(X_extended, x_batch, axis = 0)
            break
        
        # Fill labels for added images set to current class.
        added = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((added), c, dtype = int))
    
    # Print a newline to finish the progress display
    print()
    
    # Convert back to one-hot encoding if input was one-hot encoded
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_extended_one_hot = np.eye(num_classes)[y_extended.astype(int)]
        return ((X_extended * 255.).astype(np.uint8), y_extended_one_hot)
    else:
        return ((X_extended * 255.).astype(np.uint8), y_extended)