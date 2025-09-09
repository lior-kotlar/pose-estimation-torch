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

def find_peaks(x):
        
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

def show_sample_channels(sample, save_directory, filename="sample_channels.png", cmap="gray"):
    """
    Show the 4 channels of a single sample (192,192,4).
    
    Args:
        sample: numpy array or torch tensor of shape (H, W, 4)
        cmap: colormap for visualization (default: gray)
    """
    file_path = os.path.join(save_directory, filename)
    # Convert torch tensor to numpy if needed
    if hasattr(sample, "detach"):
        sample = sample.detach().cpu().numpy()
    
    assert sample.shape[-1] == 4, "Expected shape (H, W, 4)"
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(sample[:, :, i], cmap=cmap)
        axes[i].set_title(f"Channel {i+1}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Sample channels saved to {file_path}")

def show_interest_points(sample, label, save_directory, filename="interest_points.png"):
    file_path = os.path.join(save_directory, filename)
    
    # Add batch dimension so find_peaks can process [B,H,W,C]
    label_batch = label[None]  # shape (1,H,W,C)

    # Find peaks -> shape [B,3,C]
    peaks = find_peaks(label_batch)  # (x, y, val)
    coords = peaks[:, :2, :].transpose(0, 2, 1)[0]  # shape (num_points, 2)

    # Select the second channel of the sample as background
    frame = sample[:, :, 1]  

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(frame, cmap="gray")
    plt.scatter(coords[:, 0], coords[:, 1], c="red", s=40, marker="x")
    plt.title("Interest Points")

    # Save figure
    plt.savefig(file_path)
    plt.close()
    print(f"Interest points plot saved to {file_path}")

def show_interest_points_with_index(sample, label, save_directory, filename="interest_points_index.png"):
    """
    Save a plot showing the peaks of confidence maps on the second channel of the sample.
    Each peak is plotted in a different color with its index next to it.

    Args:
        sample: np.ndarray of shape (H, W, 4)
        label: np.ndarray of shape (H, W, num_points)
        save_path: str, path to save the plot
    """
    file_path = os.path.join(save_directory, filename)
    # Add batch dim for find_peaks
    label_batch = label[None]  # shape (1,H,W,C)

    # Find peaks -> shape [B,3,C]
    peaks = find_peaks(label_batch)  # returns (x, y, val)
    coords = peaks[:, :2, :].transpose(0, 2, 1)[0]  # shape (num_points, 2)

    H, W, _ = sample.shape
    num_points = coords.shape[0]

    # Second channel of sample as background
    frame = sample[:, :, 1]

    # Generate a distinct color for each point
    colors = plt.cm.get_cmap('tab10', num_points).colors  # tab10 has 10 distinct colors

    plt.figure(figsize=(6, 6))
    plt.imshow(frame, cmap='gray')

    for i in range(num_points):
        x, y = coords[i]
        plt.scatter(x, y, color=colors[i % len(colors)], s=60, marker='x')
        plt.text(x + 1, y + 1, str(i), color=colors[i % len(colors)], fontsize=12)

    plt.title("Interest Points with Indices")
    plt.axis('off')
    plt.tight_layout()

    # Ensure directory exists

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
          