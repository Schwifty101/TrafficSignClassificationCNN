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
        Xb = self.rotate(Xb)
        Xb = self.apply_projection_transform(Xb, Xb.shape[1])
        return Xb, yb

    def rotate(self, Xb):
        batch_size = Xb.shape[0]
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            delta = 30. * self.intensity
            Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode='edge')
        return Xb

    def apply_projection_transform(self, Xb, image_size):
        d = image_size * 0.3 * self.intensity
        for i in np.random.choice(Xb.shape[0], int(Xb.shape[0] * self.p), replace=False):
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
            Xb[i] = warp(Xb[i], transform, output_shape=(image_size, image_size), order=1, mode='edge')
        return Xb