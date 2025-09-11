import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import rotate, shift, zoom, affine_transform

SAMPLE_CHANNEL_SHAPE = np.array((192, 192), dtype=np.int32)  # (H, W)

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
        self.horizontal_flip = bool(config["horizontal flip"])
        self.vertical_flip = bool(config["vertical flip"])
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
        if self.horizontal_flip:
            augmentations.append(Augmentor.HorizontalFlip())
        if self.vertical_flip:
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
        def __init__(self, scale_range=(0.75, 1.25)):
            """
            Random scaling transform for image + confidence maps.

            Args:
                scale_range: tuple (min_scale, max_scale), scaling factor range
            """
            self.scale_range = scale_range

        def scale_image(self, image, scale_factor):
            
            H, W = image.shape[1:]

            zoomed_image = zoom(image, zoom=(1, scale_factor, scale_factor), order=3, mode='nearest', cval=0.0, grid_mode=True)

            zoomed_image = np.clip(zoomed_image, 0.0, 1.0)

            print(f'image shape: {image.shape}, zoomed shape: {zoomed_image.shape}, scale factor: {scale_factor}')
            return zoomed_image

        def center_images(self, scaled, scale_factor):
            center_original = (SAMPLE_CHANNEL_SHAPE // 2).astype(np.int32)
            scaled_center = (center_original * scale_factor).astype(np.int32)
            if scale_factor > 1.0:
                start = scaled_center - center_original
                end = start + SAMPLE_CHANNEL_SHAPE
                centered_scaled = scaled[:, start[0]:end[0], start[1]:end[1]]
                return centered_scaled
            
            else:
                padding_width = int((SAMPLE_CHANNEL_SHAPE[0] - scaled.shape[1])//2)
                centered_scaled = np.pad(scaled, ((0,0), (padding_width, padding_width), (padding_width, padding_width)), mode='edge')
                return centered_scaled


        def __call__(self, sample, label):
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            scaled_sample = self.scale_image(sample, scale_factor)
            scaled_sample = self.center_images(scaled_sample, scale_factor)
            scaled_label = self.scale_image(label, scale_factor)
            scaled_label = self.center_images(scaled_label, scale_factor)
            return scaled_sample, scaled_label