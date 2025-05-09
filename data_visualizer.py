import matplotlib.pyplot as plt
import numpy as np
import random
from data_loader import load_pickled_data, load_sign_names
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

def visualize_dataset(file_path, num_samples=10, random_state=None):
    """
    Visualize a random sample of images from the dataset.
    
    Args:
        file_path: Path to the pickle file containing the dataset
        num_samples: Number of images to display
        random_state: Random seed for reproducibility
    """
    # Load the data
    X, y = load_pickled_data(file_path, ['features', 'labels'])
    sign_names = load_sign_names()
    
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
    
    # Get random indices
    if len(y.shape) > 1:  # One-hot encoded
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    indices = random.sample(range(len(X)), min(num_samples, len(X)))
    
    # Create the figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Display each image
    for i, idx in enumerate(indices):
        if i < len(axes):
            # Get the image and its label
            img = X[idx]
            label = y_indices[idx]
            
            # If image is grayscale with shape (h, w, 1), convert to (h, w)
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = img.reshape(img.shape[0], img.shape[1])
                
            # Display the image
            if len(img.shape) == 2:  # Grayscale
                axes[i].imshow(img, cmap='gray')
            else:  # Color
                axes[i].imshow(img)
                
            # Add the label as title
            axes[i].set_title(f"Class {label}: {sign_names[label]}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_class_distribution(file_path):
    """
    Visualize the class distribution in the dataset.
    
    Args:
        file_path: Path to the pickle file containing the dataset
    """
    # Load the data
    _, y = load_pickled_data(file_path, ['features', 'labels'])
    sign_names = load_sign_names()
    
    # Convert one-hot encoded labels to indices if necessary
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    
    # Count instances per class
    classes, counts = np.unique(y, return_counts=True)
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'Class': [sign_names[c] for c in classes],
        'Count': counts,
        'Class_ID': classes
    })
    
    # Sort by class ID
    df = df.sort_values('Class_ID')
    
    # Visualize
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='Class_ID', y='Count', data=df)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('Traffic Sign Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in the Dataset')
    
    # Add class counts as text labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.show()

def visualize_preprocessed_vs_original(file_path, indices=None, num_samples=5):
    """
    Compare original and preprocessed images side by side.
    
    Args:
        file_path: Path to the pickle file containing the dataset
        indices: Specific indices to visualize, if None, random samples will be chosen
        num_samples: Number of samples to display if indices is None
    """
    # Load the data
    X, y = load_pickled_data(file_path, ['features', 'labels'])
    sign_names = load_sign_names()
    
    # Get indices if not provided
    if indices is None:
        indices = random.sample(range(len(X)), min(num_samples, len(X)))
    
    # Convert labels to indices if one-hot encoded
    if len(y.shape) > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Create figure
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 2*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, 2)
    
    # For each sample
    for i, idx in enumerate(indices):
        img = X[idx]
        label = y_indices[idx]
        
        # Original image
        if len(img.shape) == 3 and img.shape[2] == 1:
            # Grayscale image with channel
            axes[i, 0].imshow(img.reshape(img.shape[0], img.shape[1]), cmap='gray')
        elif len(img.shape) == 2:
            # Grayscale image without channel
            axes[i, 0].imshow(img, cmap='gray')
        else:
            # Color image
            axes[i, 0].imshow(img)
        
        axes[i, 0].set_title(f"Original Image")
        axes[i, 0].axis('off')
        
        # Preprocessed image (simulated by applying equalization)
        from skimage import exposure, util
        if len(img.shape) == 3 and img.shape[2] == 1:
            # Grayscale image with channel
            # Ensure image is in proper range before equalization
            img_norm = img.reshape(img.shape[0], img.shape[1])
            if img_norm.dtype == np.float32 or img_norm.dtype == np.float64:
                if img_norm.max() > 1.0:
                    img_norm = img_norm / 255.0  # Normalize to [0,1]
            preprocessed = exposure.equalize_adapthist(img_norm)
            axes[i, 1].imshow(preprocessed, cmap='gray')
        elif len(img.shape) == 2:
            # Grayscale image without channel
            # Ensure image is in proper range before equalization
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() > 1.0:
                    img_norm = img / 255.0  # Normalize to [0,1]
                else:
                    img_norm = img.copy()
            else:
                img_norm = img / 255.0
            preprocessed = exposure.equalize_adapthist(img_norm)
            axes[i, 1].imshow(preprocessed, cmap='gray')
        else:
            # Color image - convert to grayscale then equalize
            # Ensure image is in proper range before conversion
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() > 1.0:
                    img_norm = img / 255.0  # Normalize to [0,1]
                else:
                    img_norm = img.copy()
            else:
                img_norm = img / 255.0
            gray = 0.299 * img_norm[:, :, 0] + 0.587 * img_norm[:, :, 1] + 0.114 * img_norm[:, :, 2]
            preprocessed = exposure.equalize_adapthist(gray)
            axes[i, 1].imshow(preprocessed, cmap='gray')
        
        axes[i, 1].set_title(f"Class {label}: {sign_names[label]}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_augmented_samples(X, y, aug_intensity=0.75, num_samples=5):
    """
    Visualize original samples alongside their augmented versions.
    
    Args:
        X: Original images
        y: Labels for the images
        aug_intensity: Intensity of augmentation
        num_samples: Number of samples to display
    """
    from utils.AugmentedSignsBatchIterator import AugmentedSignsBatchIterator
    sign_names = load_sign_names()
    
    # Convert labels to indices if one-hot encoded
    if len(y.shape) > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Choose random samples
    indices = random.sample(range(len(X)), min(num_samples, len(X)))
    X_samples = X[indices]
    y_samples = y_indices[indices]
    
    # Create augmented versions
    batch_iterator = AugmentedSignsBatchIterator(batch_size=len(indices), p=1.0, intensity=aug_intensity)
    for x_augmented, _ in batch_iterator(X_samples, y_samples):
        # We need just one batch for visualization
        break
    
    # Debug information
    print(f"Original images - Min: {X_samples.min():.4f}, Max: {X_samples.max():.4f}, Shape: {X_samples.shape}, dtype: {X_samples.dtype}")
    print(f"Augmented images - Min: {x_augmented.min():.4f}, Max: {x_augmented.max():.4f}, Shape: {x_augmented.shape}, dtype: {x_augmented.dtype}")
    
    # Normalize for visualization if needed
    X_display = X_samples.copy()
    x_augmented_display = x_augmented.copy()
    
    if X_display.max() > 1:
        X_display = X_display / 255.0
    if x_augmented_display.max() > 1:
        x_augmented_display = x_augmented_display / 255.0
    
    # Create figure
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 2*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, 2)
    
    # For each sample
    for i in range(len(indices)):
        # Original image
        if len(X_display[i].shape) == 3 and X_display[i].shape[2] == 1:
            # Grayscale image with channel
            axes[i, 0].imshow(X_display[i].reshape(X_display[i].shape[0], X_display[i].shape[1]), cmap='gray')
        elif len(X_display[i].shape) == 2:
            # Grayscale image without channel
            axes[i, 0].imshow(X_display[i], cmap='gray')
        else:
            # Color image
            axes[i, 0].imshow(X_display[i])
        
        axes[i, 0].set_title(f"Original (Class {y_samples[i]})")
        axes[i, 0].axis('off')
        
        # Augmented image
        if len(x_augmented_display[i].shape) == 3 and x_augmented_display[i].shape[2] == 1:
            # Grayscale image with channel
            axes[i, 1].imshow(x_augmented_display[i].reshape(x_augmented_display[i].shape[0], x_augmented_display[i].shape[1]), cmap='gray')
        elif len(x_augmented_display[i].shape) == 2:
            # Grayscale image without channel
            axes[i, 1].imshow(x_augmented_display[i], cmap='gray')
        else:
            # Color image
            axes[i, 1].imshow(x_augmented_display[i])
        
        axes[i, 1].set_title(f"Augmented (Class {y_samples[i]})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Augmentation with intensity={aug_intensity}", y=1.02)
    plt.show()

def plot_tsne_visualization(X, y, perplexity=30, n_iter=1000, sample_size=1000):
    """
    Create a t-SNE visualization of the dataset features.
    
    Args:
        X: Feature vectors
        y: Labels
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        sample_size: Number of samples to use (for speed)
    """
    # Convert labels to indices if one-hot encoded
    if len(y.shape) > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Sample data if needed
    if sample_size < len(X):
        indices = random.sample(range(len(X)), sample_size)
        X_sample = X[indices]
        y_sample = y_indices[indices]
    else:
        X_sample = X
        y_sample = y_indices
    
    # Reshape data if needed
    if len(X_sample.shape) > 2:
        rows, cols = X_sample.shape[1], X_sample.shape[2]
        channels = X_sample.shape[3] if len(X_sample.shape) > 3 else 1
        X_sample = X_sample.reshape(X_sample.shape[0], rows * cols * channels)
    
    # Normalize
    X_sample = X_sample / 255.0 if X_sample.max() > 1 else X_sample
    
    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Create colormap with enough colors
    num_classes = len(np.unique(y_sample))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    cmap = ListedColormap(colors)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap=cmap, alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE visualization of traffic sign images (perplexity={perplexity})')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    # Add legend
    handles, labels = scatter.legend_elements()
    class_names = load_sign_names()
    legend_labels = [f"Class {i}: {class_names[i]}" for i in range(len(class_names)) if i in y_sample]
    plt.legend(handles, legend_labels, loc="best", title="Classes")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    print("Data Visualization Tools")
    print("1. Visualize random dataset samples")
    print("2. Show class distribution")
    print("3. Compare original vs preprocessed images")
    print("4. Visualize augmented samples")
    print("5. t-SNE visualization")
    
    choice = input("Select a visualization (1-5): ")
    
    if choice == '1':
        dataset = input("Enter dataset file path (default: data/train.p): ") or "data/train.p"
        visualize_dataset(dataset, num_samples=10)
    
    elif choice == '2':
        dataset = input("Enter dataset file path (default: data/train.p): ") or "data/train.p"
        visualize_class_distribution(dataset)
    
    elif choice == '3':
        dataset = input("Enter dataset file path (default: data/train.p): ") or "data/train.p"
        visualize_preprocessed_vs_original(dataset, num_samples=5)
    
    elif choice == '4':
        dataset = input("Enter dataset file path (default: data/train.p): ") or "data/train.p"
        X, y = load_pickled_data(dataset, ['features', 'labels'])
        intensity = float(input("Enter augmentation intensity (0.0-1.0, default: 0.75): ") or "0.75")
        visualize_augmented_samples(X, y, aug_intensity=intensity)
    
    elif choice == '5':
        dataset = input("Enter dataset file path (default: data/train.p): ") or "data/train.p"
        X, y = load_pickled_data(dataset, ['features', 'labels'])
        perplexity = int(input("Enter t-SNE perplexity (default: 30): ") or "30")
        sample_size = int(input("Enter sample size (default: 1000): ") or "1000")
        plot_tsne_visualization(X, y, perplexity=perplexity, sample_size=sample_size)
    
    else:
        print("Invalid choice. Please run again and select a number between 1 and 5.")