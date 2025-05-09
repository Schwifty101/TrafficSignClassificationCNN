# utils/AugmentedSignsBatchIterator.py
import numpy as np
import random
from skimage.transform import rotate, warp, ProjectiveTransform

class AugmentedSignsBatchIterator:
    def __init__(self, batch_size, shuffle=False, seed=42, p=0.5, intensity=0.5):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.p = p
        self.intensity = intensity
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self, X, y):
        return self._get_batches(X, y)

    def _get_batches(self, X, y):
        indices = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            yield self.transform(X_batch, y_batch)

    def transform(self, Xb, yb):
        Xb, yb = self._transform(Xb, yb)
        return Xb, yb

    def _transform(self, Xb, yb):
        """
        Optimized transformation pipeline that processes whole batches at once where possible
        and properly handles value ranges.
        """
        # Make a copy to avoid modifying the original data
        Xb_augmented = Xb.copy()
        
        # Check if images are normalized [0,1] or in [0,255] range
        is_normalized = Xb.max() <= 1.0
        max_val = 1.0 if is_normalized else 255.0
        
        # Apply rotations to the entire batch at once
        Xb_augmented = self.rotate_batch(Xb_augmented, max_val)
        
        # Apply projection transforms
        Xb_augmented = self.apply_projection_transform_batch(Xb_augmented, max_val)
        
        # Apply additional transformations based on intensity
        if self.intensity > 0.3:
            # Apply brightness changes
            Xb_augmented = self.adjust_brightness_batch(Xb_augmented, max_val)
            
        if self.intensity > 0.5:
            # Apply blur or sharpening
            Xb_augmented = self.adjust_sharpness_batch(Xb_augmented, max_val)
            
        if self.intensity > 0.7:
            # Apply noise
            Xb_augmented = self.add_noise_batch(Xb_augmented, max_val)
        
        return Xb_augmented, yb

    def rotate_batch(self, Xb, max_val):
        """Optimized version that processes entire batch at once."""
        batch_size = Xb.shape[0]
        delta = 45.0 * self.intensity  # Max rotation angle
        
        # Select images to transform
        indices = np.random.choice(batch_size, int(batch_size * self.p), replace=False)
        
        # Apply rotation to selected images
        for i in indices:
            # Normalize if needed
            need_normalization = max_val > 1.0
            img = Xb[i] / max_val if need_normalization else Xb[i]
            
            # Apply rotation
            angle = random.uniform(-delta, delta)
            rotated = rotate(img, angle, mode='edge')
            
            # Convert back to original range if needed
            if need_normalization:
                rotated = rotated * max_val
                
            Xb[i] = rotated
            
        return Xb

    def apply_projection_transform_batch(self, Xb, max_val):
        """Optimized version for applying projection transforms."""
        batch_size = Xb.shape[0]
        image_size = Xb.shape[1]  # Assuming square images
        
        # Scale distortion based on intensity
        d = image_size * 0.4 * self.intensity
        
        # Select images to transform
        indices = np.random.choice(batch_size, int(batch_size * self.p), replace=False)
        
        for i in indices:
            # Normalize if needed
            need_normalization = max_val > 1.0
            img = Xb[i] / max_val if need_normalization else Xb[i]
            
            # Generate random distortion points
            tl_top = random.uniform(-d, d)
            tl_left = random.uniform(-d, d)
            bl_bottom = random.uniform(-d, d)
            bl_left = random.uniform(-d, d)
            tr_top = random.uniform(-d, d)
            tr_right = random.uniform(-d, d)
            br_bottom = random.uniform(-d, d)
            br_right = random.uniform(-d, d)
            
            # Create and apply transform
            transform = ProjectiveTransform()
            transform.estimate(np.array([
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            ]), np.array([
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            ]))
            
            warped = warp(img, transform, output_shape=(image_size, image_size), order=1, mode='edge')
            
            # Convert back to original range if needed
            if need_normalization:
                warped = warped * max_val
                
            Xb[i] = warped
            
        return Xb
        
    def adjust_brightness_batch(self, Xb, max_val):
        """Optimized brightness adjustment for batches."""
        batch_size = Xb.shape[0]
        brightness_range = 0.4 * self.intensity
        
        # Select images to transform
        indices = np.random.choice(batch_size, int(batch_size * self.p), replace=False)
        
        for i in indices:
            # Randomly brighten or darken the image
            factor = 1.0 + random.uniform(-brightness_range, brightness_range)
            
            # Apply brightness adjustment (ensure we don't go out of bounds)
            Xb[i] = np.clip(Xb[i] * factor, 0, max_val)
        
        return Xb
        
    def adjust_sharpness_batch(self, Xb, max_val):
        """Optimized sharpness/blur adjustment for batches."""
        from scipy import ndimage
        batch_size = Xb.shape[0]
        
        # Select images to transform
        indices = np.random.choice(batch_size, int(batch_size * self.p), replace=False)
        
        for i in indices:
            # Decide whether to blur or sharpen
            if random.random() < 0.5:
                # Apply Gaussian blur
                sigma = random.uniform(0, 1.0 * self.intensity)
                if len(Xb[i].shape) == 3:  # RGB image
                    for c in range(Xb[i].shape[2]):
                        Xb[i][:,:,c] = ndimage.gaussian_filter(Xb[i][:,:,c], sigma=sigma)
                else:  # Grayscale
                    Xb[i] = ndimage.gaussian_filter(Xb[i], sigma=sigma)
            else:
                # Apply sharpening using unsharp mask
                blurred = ndimage.gaussian_filter(Xb[i], sigma=1.0 * self.intensity)
                filter_strength = 1.0 + 1.5 * self.intensity
                Xb[i] = Xb[i] + (Xb[i] - blurred) * filter_strength
                
                # Ensure values are within bounds
                Xb[i] = np.clip(Xb[i], 0, max_val)
                    
        return Xb
                
    def add_noise_batch(self, Xb, max_val):
        """Optimized noise addition for batches."""
        batch_size = Xb.shape[0]
        
        # Select images to transform
        indices = np.random.choice(batch_size, int(batch_size * self.p), replace=False)
        
        for i in indices:
            # Determine noise scale based on image range and intensity
            if max_val <= 1.0:
                noise_scale = 0.1 * self.intensity  # For [0,1] range
            else:
                noise_scale = 25.0 * self.intensity  # For [0,255] range
            
            # Generate noise
            noise = np.random.normal(0, noise_scale, Xb[i].shape)
            
            # Add noise to image
            Xb[i] = np.clip(Xb[i] + noise, 0, max_val)
                
        return Xb