import numpy as np
import matplotlib.pyplot as plt
import os
from Datasets import *

loss_from_string = {
    "MSE": torch.nn.MSELoss,
    "BCE": torch.nn.BCELoss,
    "BCEWithLogits": torch.nn.BCEWithLogitsLoss,
    "CrossEntropy": torch.nn.CrossEntropyLoss,
}

optimizer_from_string = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}

def tf_format_find_peaks(x):

        b, h, w, c = x.shape

        flattened = x.reshape(b, h * w, c)

        idx = np.argmax(flattened, axis=1)  # [batch, channels]

        # Convert flat index to (row, col)
        rows = idx // w
        cols = idx % w

        # Max values per channel
        vals = np.max(flattened, axis=1)  # [batch, channels]

        # Stack results into shape [batch, 3, channels]
        pred = np.stack([cols.astype(float), rows.astype(float), vals], axis=1)

        return pred

def find_peaks(x):
    """
    Find peak locations in confidence maps.

    Args:
        x: np.ndarray of shape (B, C, H, W)

    Returns:
        pred: np.ndarray of shape (B, 3, C), where for each channel:
              [0,:] = x-coords (cols),
              [1,:] = y-coords (rows),
              [2,:] = peak values.
    """
    b, c, h, w = x.shape

    # Flatten spatial dimensions: (B, C, H*W)
    flattened = x.reshape(b, c, h * w)

    # Indices of maxima along spatial dim
    idx = np.argmax(flattened, axis=2)  # [B, C]

    # Convert flat index back to (row, col)
    rows = idx // w
    cols = idx % w

    # Max values per channel
    vals = np.max(flattened, axis=2)  # [B, C]

    # Stack results → shape (B, 3, C)
    pred = np.stack([cols.astype(float), rows.astype(float), vals], axis=1)

    return pred
    

def show_sample_channels(sample, save_directory, filename="sample_channels.png", cmap="gray"):
    """
    Show the 4 channels of a single sample (4, H, W).
    
    Args:
        sample: numpy array or torch tensor of shape (4, H, W)
        cmap: colormap for visualization (default: gray)
    """
    file_path = os.path.join(save_directory, filename)
    # Convert torch tensor to numpy if needed
    if hasattr(sample, "detach"):
        sample = sample.detach().cpu().numpy()
    
    assert sample.shape[0] == 4, "Expected shape (4, H, W)"
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(sample[i, :, :], cmap=cmap)
        axes[i].set_title(f"Channel {i+1}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Sample channels saved to {file_path}")

def show_interest_points_with_index(sample, label, save_directory, filename="interest_points_index.png"):
    """
    Save a plot showing the peaks of confidence maps on the second channel of the sample.
    Each peak is plotted in a different color with its index next to it.

    Args:
        sample: np.ndarray or torch.Tensor of shape (4, H, W)
        label: np.ndarray or torch.Tensor of shape (num_points, H, W)
    """
    file_path = os.path.join(save_directory, filename)

    # Convert torch tensors to numpy
    if hasattr(sample, "detach"):
        sample = sample.detach().cpu().numpy()
    if hasattr(label, "detach"):
        label = label.detach().cpu().numpy()

    # Add batch dimension for find_peaks -> (1, num_points, H, W)
    label_batch = label[None]

    # Find peaks -> shape [B,3,C]
    peaks = find_peaks(label_batch)  # (x, y, val)
    coords = peaks[:, :2, :].transpose(0, 2, 1)[0]  # shape (num_points, 2)

    _, H, W = sample.shape
    num_points = coords.shape[0]

    # Use the 2nd channel (index 1) of sample as background
    frame = sample[1, :, :]

    # Generate a distinct color for each point
    colors = plt.cm.get_cmap('tab10', num_points).colors

    plt.figure(figsize=(6, 6))
    plt.imshow(frame, cmap='gray')

    for i in range(num_points):
        x, y = coords[i]
        plt.scatter(x, y, color=colors[i % len(colors)], s=60, marker='x')
        plt.text(x + 1, y + 1, str(i), color=colors[i % len(colors)], fontsize=12)

    plt.title("Interest Points with Indices")
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(file_path)
    plt.close()
    print(f"Indexed interest points plot saved to {file_path}")

def test_transforms(sample, label, save_directory, transforms):
     for transform in transforms:
          transform_name = transform.__class__.__name__
          transformed_sample, transformed_label = transform(sample, label)
          print(f"Applied {transform_name}")
          show_sample_channels(sample=sample, save_directory=save_directory, filename=f"original_sample_channels.png")
          show_sample_channels(sample=transformed_sample, save_directory=save_directory, filename=f"{transform_name}_sample_channels.png")
          show_interest_points_with_index(
               sample=sample,
               label=label,
               save_directory=save_directory,
               filename=f"original_interest_points_index.png")
          show_interest_points_with_index(
               sample=transformed_sample,
               label=transformed_label,
               save_directory=save_directory,
               filename=f"{transform_name}_interest_points_index.png")
          