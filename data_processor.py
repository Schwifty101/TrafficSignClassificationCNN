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
    Optimized version with pre-allocated arrays and more efficient processing.
    """
    num_classes = 43
    
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Get class counts
    _, class_counts = np.unique(y_indices, return_counts=True)
    
    # Define max count per class (either the max from existing or from provided counts)
    if counts is None:
        max_c = max(class_counts)
        target_counts = np.full(num_classes, max_c, dtype=int)
    else:
        target_counts = counts
    
    # Calculate total dataset size after balancing
    total_samples = np.sum(target_counts)
    
    # Pre-allocate the entire output arrays
    X_extended = np.zeros((total_samples, X.shape[1], X.shape[2], X.shape[3]), dtype=np.float32)
    y_extended = np.zeros(total_samples, dtype=y_indices.dtype)
    
    print(f"Extending dataset using augmented data (intensity = {aug_intensity}):")
    print(f"Target dataset size: {total_samples} samples across {num_classes} classes")
    
    # Import augmentation class now (once) instead of inside the loop
    from utils.AugmentedSignsBatchIterator import AugmentedSignsBatchIterator
    
    # Progress tracking
    progress_total = num_classes
    
    # Track the position in the output arrays
    current_position = 0
    
    # Process each class
    for c, c_count in zip(range(num_classes), class_counts):
        # Progress display
        progress_percentage = (c + 1) / progress_total * 100
        progress_bar = '█' * int(progress_percentage / 2) + '-' * (50 - int(progress_percentage / 2))
        print(f"\r[{progress_bar}] {progress_percentage:.1f}% - Processing class {c}/{num_classes-1} ", end='', flush=True)
        
        # Get current class data
        class_mask = (y_indices == c)
        X_class = X[class_mask]
        y_class = y_indices[class_mask]
        
        # Determine how many samples needed for this class
        target_count = target_counts[c]
        
        # First copy original samples
        samples_to_copy = min(c_count, target_count)
        X_extended[current_position:current_position + samples_to_copy] = X_class[:samples_to_copy] / 255.0
        y_extended[current_position:current_position + samples_to_copy] = c
        current_position += samples_to_copy
        
        # Determine remaining samples needed
        remaining = target_count - samples_to_copy
        
        if remaining > 0:
            # Use augmentation to generate the rest
            batch_size = min(c_count, 1000)  # Process in batches of max 1000 to avoid memory issues
            
            # Create the augmentation batch iterator (more efficient with a fixed batch size)
            batch_iterator = AugmentedSignsBatchIterator(
                batch_size=batch_size, 
                p=1.0,  # Apply to all images
                intensity=aug_intensity
            )
            
            # Generate augmented samples until we meet the target
            augmented_count = 0
            while augmented_count < remaining:
                # Determine how many samples to generate in this batch
                batch_to_generate = min(remaining - augmented_count, batch_size)
                
                # Get the source samples (reuse original samples if needed)
                source_indices = np.random.choice(c_count, batch_to_generate, replace=(batch_to_generate > c_count))
                X_source = X_class[source_indices] / 255.0
                y_source = y_class[source_indices]
                
                # Generate augmented batch
                for x_batch, _ in batch_iterator(X_source, y_source):
                    # Copy to the pre-allocated array
                    X_extended[current_position:current_position + batch_to_generate] = x_batch
                    y_extended[current_position:current_position + batch_to_generate] = c
                    current_position += batch_to_generate
                    augmented_count += batch_to_generate
                    # We only need one batch per iteration
                    break
    
    # Print a newline to finish the progress display
    print()
    print(f"Dataset extension complete: {current_position} total samples created")
    
    # Make sure we only return the filled portion
    X_extended = X_extended[:current_position]
    y_extended = y_extended[:current_position]
    
    # Convert back to one-hot encoding if input was one-hot encoded
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_extended_one_hot = np.eye(num_classes)[y_extended.astype(int)]
        return ((X_extended * 255.).astype(np.uint8), y_extended_one_hot)
    else:
        return ((X_extended * 255.).astype(np.uint8), y_extended)