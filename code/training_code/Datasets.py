import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import rotate, shift, zoom, affine_transform



class Dataset(Dataset):
    def __init__(self, input_samples, confmap, transforms=None):
        super().__init__()
        self.input_samples = torch.tensor(input_samples, dtype=torch.float32)
        self.confmap = torch.tensor(confmap, dtype=torch.float32)
        self.transform = transforms

    def __len__(self):
        return self.input_samples.shape[0]

    def __getitem__(self, idx):
        sample = self.input_samples[idx]
        confmap = self.confmap[idx]
        if self.transform:
            sample, confmap = self.transform(sample, confmap)
        return sample, confmap
    

class Augmentor():
    def __init__(self,
                 config):
        self.rotation_range = config["rotation range"]
        self.seed = config["seed"]
        self.zoom_range = config["zoom range"]
        self.HorizontalFlip = bool(config["horizontal flip"])
        self.VerticalFlip = bool(config["vertical flip"])
        self.shift = config["xy shift"]
        self.debug_mode = bool(config["debug mode"])
        self.batch_size = config["batch size"] if not self.debug_mode else 1
        self.augmentations = self.config_augmentations()

    def config_augmentations(self):
        augmentations = []
        if self.rotation_range > 0:
            augmentations.append(Augmentor.Rotation(rotation_range=self.rotation_range))
        if self.zoom_range is not None:
            augmentations.append(Augmentor.Scale(scale_range=self.zoom_range))
        if self.HorizontalFlip:
            augmentations.append(Augmentor.HorizontalFlip())
        if self.VerticalFlip:
            augmentations.append(Augmentor.VerticalFlip())
        if self.shift > 0:
            augmentations.append(Augmentor.Shift(shift=self.shift))
        
        return augmentations
    
    def get_transforms(self):
        double_compose = self.ComposeDouble(self.augmentations)
        return double_compose
    
    def get_augmentations(self):
        return self.augmentations
    
    class ComposeDouble:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, sample, confmap):
            for t in self.transforms:
                sample, confmap = t(sample, confmap)
            return sample, confmap

    class HorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p
        
        def __call__(self, sample, label):
            if np.random.rand() < self.p:
                # Flip width axis (horizontal flip)
                flipped_sample = np.flip(sample, axis=1).copy()
                flipped_label = np.flip(label, axis=1).copy()

                return flipped_sample, flipped_label
            else:
                return sample, label
        
    class VerticalFlip:
        def __init__(self, p=0.5):
            self.p = p
        
        def __call__(self, sample, label):
            if np.random.rand() < self.p:
                # Flip height axis (vertical flip)
                flipped_sample = np.flip(sample, axis=0).copy()
                flipped_label = np.flip(label, axis=0).copy()

                return flipped_sample, flipped_label
            else:
                return sample, label
            
    class Rotation:
        def __init__(self, rotation_range=None, angle=None):
            if rotation_range is None and angle is None:
                exit("rotration transform must receive an argument")
            self.rotation_range = rotation_range
            self.angle = angle
        
        def __call__(self, sample, label):
            angle = self.angle if self.angle is not None else np.random.uniform(-self.rotation_range, self.rotation_range)
            # Rotate sample and label
            rotated_sample = np.stack([
                rotate(sample[:, :, c], angle, reshape=False, order=1, mode='nearest')
                for c in range(sample.shape[2])
            ], axis=2)

            # Rotate all channels of label (confidence maps)
            rotated_label = np.stack([
                rotate(label[:, :, c], angle, reshape=False, order=1, mode='nearest')
                for c in range(label.shape[2])
            ], axis=2)

            return rotated_sample, rotated_label
        
    class Shift:
        def __init__(self, shift=10, range=True):
            self.range = range
            self.shift = shift

        def __call__(self, sample, label):
            
            # Random shifts in x and y
            x_shift = np.random.uniform(-self.shift, self.shift) if self.range else self.shift
            y_shift = np.random.uniform(-self.shift, self.shift) if self.range else self.shift

            # Shift all channels of sample
            shifted_sample = np.stack([
                shift(sample[:, :, c], shift=(y_shift, x_shift), order=1, mode='nearest')
                for c in range(sample.shape[2])
            ], axis=2)

            # Shift all channels of label (confidence maps)
            shifted_label = np.stack([
                shift(label[:, :, c], shift=(y_shift, x_shift), order=1, mode='nearest')
                for c in range(label.shape[2])
            ], axis=2)

            return shifted_sample, shifted_label

    class Scale:
        def __init__(self, scale_range=(0.8, 1.2)):
            """
            Random scaling transform for image + confidence maps.

            Args:
                scale_range: tuple (min_scale, max_scale), scaling factor range
            """
            self.scale_range = scale_range

        def __call__(self, sample, label):
            """
            Args:
                sample: np.ndarray, shape (H, W, C)
                label: np.ndarray, shape (H, W, num_points)
            Returns:
                scaled_sample, scaled_label
            """
            # Random scale factor
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            H, W = sample.shape[1:]
            center_y, center_x = H / 2, W / 2

            # Coordinates in the output image
            y_indices, x_indices = np.indices((H, W))

            # Map output coordinates to input coordinates (centered scaling)
            y_src = (y_indices - center_y) / scale_factor + center_y
            x_src = (x_indices - center_x) / scale_factor + center_x

            # Clip to input boundaries
            y_src_clipped = np.clip(y_src, 0, H - 1).astype(int)
            x_src_clipped = np.clip(x_src, 0, W - 1).astype(int)

            # Apply scaling
            scaled_sample = np.stack([
                sample[c, y_src_clipped, x_src_clipped]
                for c in range(sample.shape[0])
            ], axis=0)
            
            scaled_label = np.stack([
                label[c, y_src_clipped, x_src_clipped]
                for c in range(label.shape[0])
            ], axis=0)

            return scaled_sample, scaled_label
    