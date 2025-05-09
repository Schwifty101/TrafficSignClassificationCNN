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
        # Make a copy to avoid modifying the original data
        Xb_augmented = Xb.copy()
        
        # Apply rotations
        Xb_augmented = self.rotate(Xb_augmented)
        
        # Apply projection transforms
        Xb_augmented = self.apply_projection_transform(Xb_augmented, Xb_augmented.shape[1])
        
        # Apply additional transformations based on intensity
        if self.intensity > 0.3:
            # Apply brightness changes
            Xb_augmented = self.adjust_brightness(Xb_augmented)
            
        if self.intensity > 0.5:
            # Apply blur or sharpening
            Xb_augmented = self.adjust_sharpness(Xb_augmented)
            
        if self.intensity > 0.7:
            # Apply noise
            Xb_augmented = self.add_noise(Xb_augmented)
        
        return Xb_augmented, yb

    def rotate(self, Xb):
        batch_size = Xb.shape[0]
        # Increase the rotation range based on intensity
        # At intensity=1.0, this will allow rotations up to 45 degrees
        delta = 45.0 * self.intensity  
        
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            # Remember the original range
            orig_min, orig_max = Xb[i].min(), Xb[i].max()
            is_int_type = Xb[i].dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32]
            
            # Normalize to [0,1] for skimage functions if needed
            img_normalized = Xb[i]
            if orig_max > 1.0:
                img_normalized = Xb[i] / 255.0
                
            # Apply rotation with a random angle within the range
            angle = random.uniform(-delta, delta)
            rotated = rotate(img_normalized, angle, mode='edge')
            
            # Restore original range
            if orig_max > 1.0:
                rotated = rotated * 255.0
                if is_int_type:
                    rotated = np.clip(rotated, 0, 255).astype(Xb[i].dtype)
                    
            Xb[i] = rotated
            
        return Xb

    def apply_projection_transform(self, Xb, image_size):
        # Scale the distortion based on intensity
        # At intensity=1.0, distortion can be up to 40% of image size
        d = image_size * 0.4 * self.intensity
        for i in np.random.choice(Xb.shape[0], int(Xb.shape[0] * self.p), replace=False):
            # Remember the original range
            orig_min, orig_max = Xb[i].min(), Xb[i].max()
            is_int_type = Xb[i].dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32]
            
            # Normalize to [0,1] for skimage functions if needed
            img_normalized = Xb[i]
            if orig_max > 1.0:
                img_normalized = Xb[i] / 255.0
            
            tl_top = random.uniform(-d, d)
            tl_left = random.uniform(-d, d)
            bl_bottom = random.uniform(-d, d)
            bl_left = random.uniform(-d, d)
            tr_top = random.uniform(-d, d)
            tr_right = random.uniform(-d, d)
            br_bottom = random.uniform(-d, d)
            br_right = random.uniform(-d, d)
            
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
            
            warped = warp(img_normalized, transform, output_shape=(image_size, image_size), order=1, mode='edge')
            
            # Restore original range
            if orig_max > 1.0:
                warped = warped * 255.0
                if is_int_type:
                    warped = np.clip(warped, 0, 255).astype(Xb[i].dtype)
                    
            Xb[i] = warped
        return Xb
        
    def adjust_brightness(self, Xb):
        """Adjust brightness of images based on intensity parameter"""
        batch_size = Xb.shape[0]
        brightness_range = 0.4 * self.intensity
        
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            # Check the value range
            is_normalized = Xb[i].max() <= 1.0
            max_val = 1.0 if is_normalized else 255.0
            
            # Randomly brighten or darken the image
            factor = 1.0 + random.uniform(-brightness_range, brightness_range)
            
            # Apply brightness adjustment (ensure we don't go out of bounds)
            Xb[i] = np.clip(Xb[i] * factor, 0, max_val)
        
        return Xb
        
    def adjust_sharpness(self, Xb):
        """Add blur or sharpen images based on intensity parameter"""
        from scipy import ndimage
        batch_size = Xb.shape[0]
        
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            # Check the value range
            is_normalized = Xb[i].max() <= 1.0
            max_val = 1.0 if is_normalized else 255.0
            
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
                
    def add_noise(self, Xb):
        """Add random noise to images based on intensity parameter"""
        batch_size = Xb.shape[0]
        
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            # Determine noise scale based on image range and intensity
            is_normalized = Xb[i].max() <= 1.0
            
            if is_normalized:
                noise_scale = 0.1 * self.intensity  # For [0,1] range
                max_val = 1.0
            else:
                noise_scale = 25.0 * self.intensity  # For [0,255] range
                max_val = 255
            
            # Generate noise
            noise = np.random.normal(0, noise_scale, Xb[i].shape)
            
            # Add noise to image
            Xb[i] = np.clip(Xb[i] + noise, 0, max_val)
            
        return Xb