import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import rotate, shift, zoom
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import Config

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
                 general_configuration: Config):
        self.rotation_range,\
        self.zoom_range,\
        self.horizontal_flip,\
        self.vertical_flip,\
        self.shift = general_configuration.get_augmentation_configuration()
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
                flipped_sample = np.flip(sample, axis=2).copy()
                flipped_label = np.flip(label, axis=2).copy()

                return flipped_sample, flipped_label
            else:
                return sample, label
        
    class VerticalFlip:
        def __init__(self, p=0.5):
            self.p = p
        
        def __call__(self, sample, label):
            if np.random.rand() < self.p:
                # Flip height axis (vertical flip)
                flipped_sample = np.flip(sample, axis=1).copy()
                flipped_label = np.flip(label, axis=1).copy()

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

            rotated_sample = rotate(sample, angle, axes=(1,2), reshape=False, order=2, mode='nearest')
            rotated_sample = np.clip(rotated_sample, 0.0, 1.0)
            rotated_label = rotate(label, angle, axes=(1,2), reshape=False, order=2, mode='nearest')
            rotated_label = np.clip(rotated_label, 0.0, 1.0)

            return rotated_sample, rotated_label

    class Shift:
        def __init__(self, shift=10, range=True):
            self.range = range
            self.shift = shift

        def __call__(self, sample, label):
            
            # Random shifts in x and y
            x_shift = np.random.uniform(-self.shift, self.shift) if self.range else self.shift
            y_shift = np.random.uniform(-self.shift, self.shift) if self.range else self.shift

            # # Shift all channels of sample
            # shifted_sample = np.stack([
            #     shift(sample[:, :, c], shift=(y_shift, x_shift), order=1, mode='nearest')
            #     for c in range(sample.shape[2])
            # ], axis=2)

            # # Shift all channels of label (confidence maps)
            # shifted_label = np.stack([
            #     shift(label[:, :, c], shift=(y_shift, x_shift), order=1, mode='nearest')
            #     for c in range(label.shape[2])
            # ], axis=2)

            shifted_sample = shift(sample, shift=(0, y_shift, x_shift), mode='nearest')
            shifted_label = shift(label, shift=(0, y_shift, x_shift), mode='nearest')
            shifted_sample = np.clip(shifted_sample, 0.0, 1.0)
            shifted_label = np.clip(shifted_label, 0.0, 1.0)

            return shifted_sample, shifted_label
        
    class Scale:
        def __init__(self, scale_range=(0.75, 1.25)):
            """
            Random scaling transform for image + confidence maps.

            Args:
                scale_range: tuple (min_scale, max_scale), scaling factor range
            """
            self.scale_range = scale_range
        
        def scale_example(self, sample, label, scale_factor):
            
            zoomed_sample = zoom(sample, zoom=(1, scale_factor, scale_factor), order=3, mode='nearest', cval=0.0, grid_mode=True)
            zoomed_sample = np.clip(zoomed_sample, 0.0, 1.0)

            zoomed_label = zoom(label, zoom=(1, scale_factor, scale_factor), order=3, mode='nearest', cval=0.0, grid_mode=True)
            zoomed_label = np.clip(zoomed_label, 0.0, 1.0)

            # print(f'sample shape: {sample.shape}, zoomed shape: {zoomed_sample.shape}, scale factor: {scale_factor}')
            return zoomed_sample, zoomed_label
            
        def center_example(self, scaled_sample, scaled_label, scale_factor):
            center_original = (SAMPLE_CHANNEL_SHAPE // 2).astype(np.int32)
            scaled_center = (center_original * scale_factor).astype(np.int32)
            if scale_factor > 1.0:
                start = scaled_center - center_original
                end = start + SAMPLE_CHANNEL_SHAPE
                centered_scaled_sample = scaled_sample[:, start[0]:end[0], start[1]:end[1]]
                centered_scaled_label = scaled_label[:, start[0]:end[0], start[1]:end[1]]
            
            else:
                padding_width1 = int((SAMPLE_CHANNEL_SHAPE[0] - scaled_sample.shape[1])//2)
                padding_width2 = padding_width1
                fixed_width = scaled_sample.shape[1] + 2*padding_width1
                addition = SAMPLE_CHANNEL_SHAPE[0] - fixed_width
                if addition > 1:
                    exit("something is wrong with the scale size calculation")
                if addition == 1:
                    padding_width2 += 1
                centered_scaled_sample = np.pad(scaled_sample,
                                                ((0,0), (padding_width1, padding_width2), (padding_width1, padding_width2)),
                                                mode='edge')
                centered_scaled_label = np.pad(scaled_label,
                                                ((0,0), (padding_width1, padding_width2), (padding_width1, padding_width2)),
                                                mode='constant')
            
            # print(f'centered sample and label are of shape: {centered_scaled_sample.shape}, {centered_scaled_label.shape}')
            return centered_scaled_sample, centered_scaled_label


        def __call__(self, sample, label):
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            scaled_sample, scaled_label = self.scale_example(sample, label, scale_factor)
            scaled_sample, scaled_label = self.center_example(scaled_sample, scaled_label, scale_factor)
            return scaled_sample, scaled_label
        
def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset=dataset)
    )