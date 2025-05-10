# data_processor.py
import numpy as np
from sklearn.utils import shuffle
from skimage import exposure
import warnings
import time
import sys
import tensorflow as tf
from utils.AugmentedSignsBatchIterator import AugmentedSignsBatchIterator

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

def print_debug(message):
    """Print debug message with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {message}")

def preprocess_dataset(X, y=None):
    """
    Performs feature scaling, one-hot encoding of labels and shuffles the data if labels are provided.
    
    Args:
        X: Image data with shape (samples, height, width, channels)
        y: Label data (optional)
        
    Returns:
        Preprocessed X and y
    """
    print_debug(f"Preprocessing dataset with {X.shape[0]} examples")
    start_time = time.time()
    
    # Check if input data has correct shape - we always want (samples, height, width, channels)
    if len(X.shape) != 4:
        print_debug(f"Warning: Expected 4D input (batch, height, width, channels), got shape {X.shape}")
        if len(X.shape) == 3:
            # Assume this is a single-channel grayscale or RGB without channel dimension
            print_debug("Adding channel dimension")
            X = X.reshape(X.shape + (1,))
    
    # Convert to grayscale (TF 2.x compatible method)
    print_debug("Converting to grayscale...")
    try:
        # Ensure we're working with float32
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # If input has 3 channels, convert to grayscale
        if X.shape[3] == 3:
            total_examples = X.shape[0]
            batch_size = 1000  # Process in batches to show progress
            processed_images = []
            
            for batch_idx in range(0, total_examples, batch_size):
                # Get the current batch
                end_idx = min(batch_idx + batch_size, total_examples)
                batch = X[batch_idx:end_idx]
                
                # Convert batch to grayscale and keep the channel dimension
                grayscale_batch = 0.299 * batch[:, :, :, 0] + 0.587 * batch[:, :, :, 1] + 0.114 * batch[:, :, :, 2]
                # Reshape to include channel dimension
                grayscale_batch = grayscale_batch.reshape(grayscale_batch.shape + (1,))
                processed_images.append(grayscale_batch)
                
                # Show progress
                progress_bar(end_idx, total_examples, 
                            prefix="Grayscale conversion", 
                            suffix=f"Processed {end_idx}/{total_examples} images")
            
            # Combine all batches
            X = np.concatenate(processed_images, axis=0)
            print_debug(f"Grayscale conversion completed in {time.time() - start_time:.2f}s")
        else:
            print_debug(f"Input already has {X.shape[3]} channels, ensuring single-channel grayscale format")
            # Ensure we keep the channel dimension
            if X.shape[3] != 1:
                print_debug(f"Unexpected channel count: {X.shape[3]}, reshaping to single channel")
                # Reshape to ensure single channel
                X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    except Exception as e:
        print_debug(f"Error during grayscale conversion: {e}")
        # Fallback to original method
        if X.shape[3] == 3:
            X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
            # Make sure to keep channel dimension
            X = X.reshape(X.shape + (1,))
    
    # Ensure we have a channel dimension before proceeding
    print_debug(f"Current shape after grayscale conversion: {X.shape}")
    if len(X.shape) != 4:
        print_debug("Adding channel dimension for consistent processing")
        X = X.reshape(X.shape + (1,))
    
    # Scale features to be in [0, 1]
    # Check if data is already in [0, 1] range
    if X.max() > 1.0:
        print_debug("Scaling features to [0, 1] range...")
        X = (X / 255.).astype(np.float32)
    else:
        print_debug("Data already in [0, 1] range, skipping scaling...")
        # Ensure data type is float32 without scaling
        X = X.astype(np.float32)
    
    # Apply histogram equalization with progress tracking
    print_debug("Applying adaptive histogram equalization...")
    print_debug(f"Shape before equalization: {X.shape}")
    equalize_start = time.time()
    
    # Process each image while maintaining channel dimension
    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Apply equalize_adapthist to the 2D image but keep channel dimension
            X[i, :, :, 0] = exposure.equalize_adapthist(X[i, :, :, 0])
        
        # Update progress every 50 images or at the end
        if (i + 1) % 50 == 0 or i == X.shape[0] - 1:
            progress_bar(i + 1, X.shape[0], 
                        prefix="Histogram equalization", 
                        suffix=f"Processed {i+1}/{X.shape[0]} images")
    
    print_debug(f"Shape after equalization: {X.shape}")
    print_debug(f"Histogram equalization completed in {time.time() - equalize_start:.2f}s")
    
    if y is not None:
        # Convert to one-hot encoding
        print_debug("Converting labels to one-hot encoding...")
        num_classes = np.max(y) + 1
        
        # TensorFlow 2.x compatible one-hot encoding
        y = tf.one_hot(y, depth=num_classes).numpy()
        
        print_debug("Shuffling dataset...")
        X, y = shuffle(X, y)
        
        # Note: We already added the channel dimension earlier
    
    total_time = time.time() - start_time
    print_debug(f"Preprocessing completed in {total_time:.2f}s")
    return X, y

def flip_extend(X, y):
    """
    Extends existing images dataset by flipping images of some classes.
    
    Args:
        X: Image data
        y: Label data (can be one-hot encoded or class indices)
        
    Returns:
        Extended dataset (X_extended, y_extended)
    """
    print_debug(f"Starting flip-based dataset extension for {X.shape[0]} images")
    start_time = time.time()
    
    # Check if y is one-hot encoded, and convert to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        print_debug("Detected one-hot encoded labels, converting to class indices for processing")
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
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
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y_indices.dtype)
    
    print_debug(f"Processing {num_classes} classes for flip augmentation")
    
    for c in range(num_classes):
        class_start_time = time.time()
        # Show progress
        progress_bar(c + 1, num_classes, 
                    prefix="Flip augmentation", 
                    suffix=f"Processing class {c+1}/{num_classes}")
        
        class_count = np.sum(y_indices == c)
        print_debug(f"Class {c}: Found {class_count} examples")
        
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y_indices == c], axis=0)
        added_count = X_extended.shape[0] - y_extended.shape[0]
        print_debug(f"  - Added {added_count} original images")
        
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            horizontal_start = time.time()
            X_extended = np.append(X_extended, X[y_indices == c][:, :, ::-1, :], axis=0)
            added = X_extended.shape[0] - y_extended.shape[0] - added_count
            print_debug(f"  - Added {added} horizontally flipped images in {time.time() - horizontal_start:.2f}s")
            added_count += added
            
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            cross_start = time.time()
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y_indices == flip_class][:, :, ::-1, :], axis=0)
            added = X_extended.shape[0] - y_extended.shape[0] - added_count
            print_debug(f"  - Added {added} cross-flipped images from class {flip_class} in {time.time() - cross_start:.2f}s")
            added_count += added
            
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))
        
        # If we can flip images of this class vertically and they would still belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            vertical_start = time.time()
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis=0)
            added = X_extended.shape[0] - y_extended.shape[0]
            print_debug(f"  - Added {added} vertically flipped images in {time.time() - vertical_start:.2f}s")
            
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))
            
        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            both_start = time.time()
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)
            added = X_extended.shape[0] - y_extended.shape[0]
            print_debug(f"  - Added {added} horizontally+vertically flipped images in {time.time() - both_start:.2f}s")
            
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))
            
        print_debug(f"Class {c} processing completed in {time.time() - class_start_time:.2f}s")
        
    total_time = time.time() - start_time
    print_debug(f"Flip extension completed: {X.shape[0]} → {X_extended.shape[0]} images in {total_time:.2f}s")
    
    # If original labels were one-hot encoded, convert back to one-hot
    if len(y.shape) > 1 and y.shape[1] > 1:
        print_debug("Converting extended labels back to one-hot encoding format...")
        y_extended_onehot = tf.one_hot(y_extended, depth=num_classes).numpy()
        y_final = y_extended_onehot
    else:
        y_final = y_extended
    
    return (X_extended, y_final)

def extend_balancing_classes(X, y, aug_intensity=0.5, counts=None, max_total_size=None):
    """
    Extends dataset by duplicating existing images while applying data augmentation pipeline.
    
    Args:
        X: Image data
        y: Label data (can be one-hot encoded or class indices)
        aug_intensity: Augmentation intensity (0.0 to 1.0)
        counts: Target count for each class (array of 43 integers)
        max_total_size: Maximum total size of the dataset (defaults to 2x original size)
        
    Returns:
        Extended and balanced dataset (X_extended, y_extended)
    """
    num_classes = 43
    
    # Check if y is one-hot encoded, and convert to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        print_debug("Detected one-hot encoded labels, converting to class indices for processing")
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Calculate maximum total size (default to 2x original size)
    original_size = X.shape[0]
    if max_total_size is None:
        max_total_size = original_size * 2
    print_debug(f"Maximum total dataset size set to {max_total_size} (original: {original_size})")
    
    # Find current class distribution
    classes, class_counts = np.unique(y_indices, return_counts=True)
    
    # Calculate max samples per class to stay within limit
    # Default: balance all classes to the same size
    if counts is None:
        # Calculate an ideal balanced target count per class
        balanced_per_class = max_total_size // num_classes
        # But don't go below existing class counts
        counts = np.maximum(class_counts, np.full(num_classes, balanced_per_class))
    
    # Ensure we don't exceed max_total_size
    if np.sum(counts) > max_total_size:
        print_debug(f"Warning: Target class counts exceed max size, scaling down...")
        scale_factor = max_total_size / np.sum(counts)
        counts = np.ceil(counts * scale_factor).astype(int)
    
    print_debug(f"Target counts per class: min={np.min(counts)}, max={np.max(counts)}, total={np.sum(counts)}")
    total = np.sum(counts)
    
    print_debug(f"Starting dataset extension with augmentation (intensity = {aug_intensity})")
    print_debug(f"Initial dataset: {X.shape[0]} images, target: ~{total} images")
    
    start_time = time.time()
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=np.float32)
    y_extended = np.empty([0], dtype=y_indices.dtype)
    
    for c_idx, c in enumerate(range(num_classes)):
        class_start_time = time.time()
        
        # Show progress for class processing
        progress_bar(c_idx + 1, num_classes, 
                    prefix="Class augmentation", 
                    suffix=f"Processing class {c+1}/{num_classes}")
        
        # Get count for this class
        c_count = class_counts[c_idx] if c_idx < len(class_counts) else 0
        
        # How many examples should there be eventually for this class:
        target_count = counts[c] if counts is not None and c < len(counts) else c_count
        print_debug(f"Class {c}: {c_count} → {target_count} examples")
        
        # First copy existing data for this class
        class_data = X[y_indices == c]
        if len(class_data) > 0:
            print_debug(f"Class {c}: Original data type: {class_data.dtype}, min: {class_data.min()}, max: {class_data.max()}")
            # Ensure data is float32 but don't scale again - data is already normalized to [0,1] from preprocess_dataset
            X_source = class_data.astype(np.float32)
            print_debug(f"Class {c}: After conversion data type: {X_source.dtype}, min: {X_source.min()}, max: {X_source.max()}")
        else:
            print_debug(f"Class {c}: No samples found, skipping conversion")
            # Create empty array with expected data type
            X_source = np.empty((0,) + X.shape[1:], dtype=np.float32)
            
        y_source = y_indices[y_indices == c]
        X_extended = np.append(X_extended, X_source, axis=0)
        added_original = X_source.shape[0]
        
        try:
            # Import AugmentedSignsBatchIterator in a TF 2.x compatible way
            # from utils import AugmentedSignsBatchIterator
            
            # Calculate how many times we need to duplicate the class
            if c_count > 0:
                full_duplications = (target_count // c_count) - 1
                remaining = target_count % c_count
            else:
                # Skip augmentation for classes with no samples
                print_debug(f"  - Skipping augmentation for class {c} (no samples)")
                full_duplications = 0
                remaining = 0
            
            if full_duplications > 0:
                print_debug(f"  - Creating {full_duplications} full duplications of class {c}")
                
                for i in range(full_duplications):
                    dup_start = time.time()
                    batch_iterator = AugmentedSignsBatchIterator(batch_size=X_source.shape[0], p=1.0, intensity=aug_intensity)
                    
                    # Show inner progress for each duplication
                    inner_progress = 0
                    for x_batch, _ in batch_iterator(X_source, y_source):
                        X_extended = np.append(X_extended, x_batch, axis=0)
                        inner_progress += len(x_batch)
                        progress_bar(inner_progress, X_source.shape[0], 
                                    prefix=f"  Duplication {i+1}/{full_duplications}", 
                                    suffix=f"{inner_progress}/{X_source.shape[0]} images")
                    
                    print_debug(f"  - Duplication {i+1} completed in {time.time() - dup_start:.2f}s")
            
            # Handle remaining images needed
            if remaining > 0:
                remain_start = time.time()
                print_debug(f"  - Creating partial duplication with {remaining} images")
                batch_iterator = AugmentedSignsBatchIterator(batch_size=remaining, p=1.0, intensity=aug_intensity)
                
                for x_batch, _ in batch_iterator(X_source, y_source):
                    X_extended = np.append(X_extended, x_batch, axis=0)
                    break
                
                print_debug(f"  - Partial duplication completed in {time.time() - remain_start:.2f}s")
            
        except Exception as e:
            print_debug(f"Error during augmentation for class {c}: {e}")
            # Simple duplication as fallback
            needed = target_count - X_source.shape[0]
            if needed > 0:
                indices = np.random.choice(range(X_source.shape[0]), needed)
                X_extended = np.append(X_extended, X_source[indices], axis=0)
        
        # Fill labels for added images set to current class.
        added = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((added), c, dtype=int))
        
        print_debug(f"Class {c} processing completed: {added_original} → {added} images in {time.time() - class_start_time:.2f}s")
    
    # Keep as float32 to maintain precision
    print_debug("Maintaining float32 precision (not converting to uint8)...")
    
    # Check and ensure the data is in [0,1] range
    min_val, max_val = X_extended.min(), X_extended.max()
    print_debug(f"Final dataset: dtype={X_extended.dtype}, min={min_val:.6f}, max={max_val:.6f}")
    
    # Ensure data stays in [0,1] range - apply clipping if needed due to augmentation
    if max_val > 1.0 or min_val < 0.0:
        print_debug(f"Clipping values to ensure [0,1] range (found min={min_val:.6f}, max={max_val:.6f})")
        X_extended = np.clip(X_extended, 0.0, 1.0)
        print_debug(f"After clipping: min={X_extended.min():.6f}, max={X_extended.max():.6f}")
    
    # If original labels were one-hot encoded, convert back to one-hot
    if len(y.shape) > 1 and y.shape[1] > 1:
        print_debug("Converting extended labels back to one-hot encoding format...")
        y_extended_onehot = tf.one_hot(y_extended, depth=num_classes).numpy()
        y_final = y_extended_onehot
    else:
        y_final = y_extended
    
    total_time = time.time() - start_time
    print_debug(f"Dataset extension completed: {X.shape[0]} → {X_extended.shape[0]} images in {total_time:.2f}s")
    
    return (X_extended, y_final)
