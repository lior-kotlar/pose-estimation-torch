import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

OUTPUT_DIR = "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/data_analysis_output"

def preprocess(X, permute=(0, 3, 2, 1)):
    # Add singleton dim for single train_images
    if X.ndim == 3:
        X = X[None, ...]

    # Adjust dimensions
    if permute != None:
        X = np.transpose(X, permute)

    # Normalize
    if X.dtype == "uint8" or np.max(X) > 1:
        X = X.astype("float32") / 255

    return X

def load_dataset(data_path):
    """ Loads and normalizes datasets. """
    # Load
    X_dset = "box"
    Y_dset = "confmaps"
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]

    # Adjust dimensions
    X = preprocess(X, permute=None)
    Y = preprocess(Y, permute=None)
    if X.shape[0] != 2 and X.shape[1] != 4:
        X = X.T
    if Y.shape[0] != 2 or Y.shape[1] == 192:
        Y = Y.T
    return X, Y

def split_frame_to_wings(frame):
    left_mask_channels = [1, 3]
    right_mask_channels = [1, 4]
    left_wing_frame = frame[:, :, :, left_mask_channels]
    right_wing_frame = frame[:, :, :, right_mask_channels]
    return left_wing_frame, right_wing_frame

def extract_quads(box):
    examples = []
    for frame_i in range(box.shape[0]):
        frame = box[frame_i]
        left_wing_frame, right_wing_frame = split_frame_to_wings(frame)
        examples.append(left_wing_frame)
        examples.append(right_wing_frame)
    return np.array(examples)

def get_mask_area(mask):
    area = np.sum(mask)
    return int(area)

def apply_masks(images, masks):
    """
    Multiplies images by masks to set background to black.
    Assumes images are uint8 (0-255) and masks are float64 (0.0/1.0).
    """
    masked_images = []
    for i in range(len(images)):
        img = images[i]
        mask = masks[i]
        # Element-wise multiplication.
        # The result becomes float64, so we cast back to uint8 for standard image format.
        applied = (img * mask)
        masked_images.append(applied)
        
    return np.array(masked_images)

def visualize_wing_masks(masks, images, areas, ratios, masked_images, threshold):
    """
    Plots a 3x4 grid.
    Row 1: Masks + Stats
    Row 2: Source Images
    Row 3: Images with mask applied
    """
    n_cameras = 4
    # Changed to 3 rows. Increased height from 8 to 12 to fit the new row nicely.
    fig, axes = plt.subplots(3, n_cameras, figsize=(16, 12))
    
    for i in range(n_cameras):
        # --- Row 1: Masks + Geometric Stats ---
        ax_mask = axes[0, i]
        # (Same setup as before for titles and colors)
        ratio = ratios[i]
        color = 'green' if ratio > threshold else 'red'
        title_text = f"Cam {i+1} Mask\nArea: {int(areas[i])}\nA/P Ratio: {ratio:.2f}"
        
        ax_mask.imshow(masks[i], cmap='gray', vmin=0, vmax=1)
        ax_mask.set_title(title_text, fontsize=11, fontweight='bold', color=color)
        ax_mask.axis('off')

        # --- Row 2: Original Source Images ---
        ax_source = axes[1, i]
        ax_source.imshow(images[i], cmap='gray')
        ax_source.set_title(f"Cam {i+1} Source", fontsize=11)
        ax_source.axis('off')
        
        # --- Row 3: Masked Result Images (New) ---
        ax_result = axes[2, i]
        # Display the image where the background has been removed
        ax_result.imshow(masked_images[i], cmap='gray')
        ax_result.set_title(f"Cam {i+1} Applied", fontsize=11)
        ax_result.axis('off')

    plt.tight_layout()
    save_path = path.join(OUTPUT_DIR, "wing_masks_output.png")
    plt.savefig(save_path)
    plt.close(fig)

def expand_masks(masks, pixels=10):
    """
    Expands (dilates) the binary masks by a specified number of pixels.
    Input: Float64 masks (0.0/1.0)
    Output: Expanded Float64 masks
    """
    expanded_masks = []
    
    for mask in masks:
        # Start with the current mask
        current_mask = mask.copy()
        
        # Repeat the expansion for the number of pixels requested
        for _ in range(pixels):
            # Pad the image with 0s so we can check edges easily
            padded = np.pad(current_mask, pad_width=1, mode='constant', constant_values=0)
            
            # Get views of the neighbors (Up, Down, Left, Right)
            center = padded[1:-1, 1:-1]
            up     = padded[0:-2, 1:-1]
            down   = padded[2:,   1:-1]
            left   = padded[1:-1, 0:-2]
            right  = padded[1:-1, 2:]
            
            # "Maximum" logic: If the pixel OR any neighbor is 1.0, the result is 1.0
            # This effectively grows the mask by 1 pixel in all directions
            current_mask = np.maximum.reduce([center, up, down, left, right])
            
        expanded_masks.append(current_mask)
        
    return np.array(expanded_masks)

def calculate_shape_metrics(masks, areas):
    """
    Calculates the perimeter and the Area/Perimeter ratio.
    Returns:
        ratios: Array of shape (4,) representing Area / Perimeter
    """
    ratios = []
    
    for i in range(len(masks)):
        mask = masks[i]
        area = areas[i]
        
        # --- 1. Calculate Perimeter using NumPy Slicing ---
        # We check each pixel: if it's 1.0 but has a 0.0 neighbor, it's an edge.
        # We do this by shifting the array in 4 directions.
        
        # Pad the mask with 0s so we can check edges at the image border
        padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        
        # 'center' is the original mask (shifted back to align)
        center = padded[1:-1, 1:-1]
        
        # Shifted versions to check Up, Down, Left, Right neighbors
        # If ANY neighbor is 0, the bitwise '&' will fail, meaning it's an edge.
        # We want the "Eroded" version (pixels completely surrounded by 1s)
        eroded = (center == 1.0) & \
                 (padded[0:-2, 1:-1] == 1.0) & \
                 (padded[2:, 1:-1]   == 1.0) & \
                 (padded[1:-1, 0:-2] == 1.0) & \
                 (padded[1:-1, 2:]   == 1.0)
                 
        # Perimeter = Original Mask - Inner Core (Eroded)
        # This leaves only the boundary ring.
        perimeter_img = center - eroded.astype(np.float64)
        
        perimeter_count = np.sum(perimeter_img)
        
        # --- 2. Calculate Ratio (Area / Perimeter) ---
        if perimeter_count > 0:
            ratio = area / perimeter_count
        else:
            ratio = 0.0
            
        ratios.append(ratio)
        
    return np.array(ratios)

def create_area_array(masks_images):
    areas = []
    ratios = []
    for frame_i in range(masks_images.shape[0]):
        frame_masks = masks_images[frame_i,:,:,:,1]
        frame_mask_areas = [get_mask_area(frame_masks[cam, :, :]) for cam in range(frame_masks.shape[0])]
        areas.append(frame_mask_areas)
        frame_ratios = calculate_shape_metrics(frame_masks, frame_mask_areas)
        ratios.append(frame_ratios)
    return np.array(areas), np.array(ratios)

def threshold_data_analysis(ratios, threshold, masks_images, areas, visualize = False):
    how_many_above_threshold_per_frame = []
    for frame_i in range(ratios.shape[0]):
        frame_ratios = ratios[frame_i]
        count_above_threshold = np.sum(frame_ratios > threshold)
        how_many_above_threshold_per_frame.append(count_above_threshold)
        if visualize and count_above_threshold < 5:
            print(f"Frame {frame_i}: {count_above_threshold} wings above threshold")
            frame_masks = masks_images[frame_i,:,:,:,1]
            frame_masks = expand_masks(frame_masks, pixels=10)
            frame_images = masks_images[frame_i,:,:,:,0]
            frame_mask_areas = areas[frame_i]
            masked_images = apply_masks(frame_images, frame_masks)
            visualize_wing_masks(frame_masks, frame_images, frame_mask_areas, frame_ratios, masked_images, threshold)
    return how_many_above_threshold_per_frame

def plot_ratio_distribution(ratios_list, n_bins=100):
    """
    Plots a histogram of all wing ratios to help identify the separation
    between 'Side View' and 'Face View'.
    
    Args:
        ratios_list: A list or numpy array of shape (N, 4). 
                     N = number of frames, 4 = cameras per frame.
        n_bins: Number of bars in the histogram (higher = narrower bins).
    """
    # 1. Flatten the data
    # We don't care which camera it came from, we just want to see the 
    # distribution of ALL shapes to find the cut-off threshold.
    # Shape becomes (N * 4,)
    flat_data = np.array(ratios_list).flatten()
    
    # 2. Setup the Plot
    fig = plt.figure(figsize=(12, 6))
    
    # 3. Create Histogram
    # alpha=0.7 makes it slightly transparent
    # edgecolor='black' helps distinguish the narrow bins
    counts, bins, patches = plt.hist(flat_data, bins=n_bins, color='#3498db', edgecolor='black', alpha=0.7)
    
    # 4. Styling
    plt.title(f"Distribution of Wing Mask Ratios (Total Images: {len(flat_data)})", fontsize=14)
    plt.xlabel("Ratio Value (Area / Perimeter)", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    
    # Add grid only on Y axis for readability
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    
    # 5. Show Statistics
    # Draw a vertical line for the mean, just for reference
    mean_val = np.mean(flat_data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    plt.legend()

    plt.tight_layout()
    save_path = path.join(OUTPUT_DIR, "ratio_distribution.png")
    plt.savefig(save_path)
    plt.close(fig)

def plot_good_masks_histogram(pass_counts, threshold):
    """
    Plots a histogram showing how many frames had 0, 1, 2, 3, or 4 good masks.
    
    Args:
        pass_counts: A list or array of integers (e.g., [4, 3, 4, 2...]) 
                     representing the number of valid masks in each frame.
    """
    # 1. Setup Bins for Discrete Data
    # We want bars centered exactly on 0, 1, 2, 3, 4.
    # So we define edges at -0.5, 0.5, 1.5, etc.
    bins = np.arange(6) - 0.5 
    
    # 2. Plot
    fig = plt.figure(figsize=(10, 6))
    
    # rwidth=0.8 creates a small gap between bars so they don't look like a blob
    counts, edges, patches = plt.hist(pass_counts, bins=bins, color='#2ecc71', edgecolor='black', rwidth=0.8)
    
    # 3. Styling
    plt.title(f"Quality Check: How many cameras captured the wing well?\nThreshold: {threshold}", fontsize=14)
    plt.xlabel("Number of Masks Passing Threshold (0-4)", fontsize=12)
    plt.ylabel("Number of Frames", fontsize=12)
    
    # Force X-axis to only show integers 0, 1, 2, 3, 4
    plt.xticks(range(5))
    
    # Add grid for easier reading of counts
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Annotate bars with the actual count numbers (Optional but helpful)
    for count, x in zip(counts, range(5)):
        if count > 0:
            plt.text(x, count, str(int(count)), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = path.join(OUTPUT_DIR, f"good_masks_histogram_{threshold}.png")
    plt.savefig(save_path)
    plt.close(fig)

def analyze_for_thrsholds(ratios, thresholds, masks_images, areas):
    for threshold in thresholds:
        threshold_counts = threshold_data_analysis(ratios=ratios, threshold=threshold, masks_images=masks_images, areas=areas)
        plot_good_masks_histogram(threshold_counts, threshold)

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_analysis.py <data_path>")
        return
    data_path = sys.argv[1]
    X, _ = load_dataset(data_path)
    quads = extract_quads(X)
    areas, ratios = create_area_array(quads)
    plot_ratio_distribution(ratios)
    analyze_for_thrsholds(ratios, thresholds=[3.5, 4.0, 4.5, 4.7, 5.0], masks_images=quads, areas=areas)

if __name__ == "__main__":
    main()