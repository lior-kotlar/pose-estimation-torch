import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os
import shutil
import json
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop

TRAINING_CODE_DIRECTORY = "code/training_code"
SBATCH_FILES_DIRECTORY = "sbatch_files"

loss_from_string = {
    "MSE": MSELoss,
    "BCE": BCELoss,
    "BCEWithLogits": BCEWithLogitsLoss,
    "CrossEntropy": CrossEntropyLoss,
}

optimizer_from_string = {
    "Adam": Adam,
    "SGD": SGD,
    "RMSprop": RMSprop,
}

class Config:
    def __init__(self, config_path):
        with open(config_path) as CF:
            config = json.load(CF)
            # training configuration
            self.config = config
            self.batch_size = config['batch size']
            self.num_epochs = config['epochs']
            self.batches_per_epoch = config['batches per epoch']
            self.val_fraction = config['val fraction']
            self.loss_function_as_string = config["loss function"]
            self.learning_rate = config["learning rate"]
            self.optimizer_as_string = config["optimizer"]
            self.reduce_lr_factor = config["reduce lr factor"]
            self.reduce_lr_patience = config["reduce lr patience"]
            self.reduce_lr_min_delta = config["reduce lr min delta"]
            self.reduce_lr_cooldown = config["reduce lr cooldown"]
            self.reduce_lr_min_lr = config["reduce lr min lr"]
            self.base_output_directory = config["base output directory"]
            self.viz_idx = 1
            self.model_type = config["model type"]
            self.save_every = config["save every"]
            self.debug_mode = bool(config["debug mode"])
            self.confmaps_orig = None
            self.box_orig = None
            self.data_path = config['data path']
            self.test_path = config['test path']
            
            self.resume_training_checkpoint_path = config.get("training checkpoint file path", None)
            self.resume_training_directory = config.get("resume training directory", None)
            
            # preprocessing configuration
            self.mix_with_test = bool(config['mix with test'])
            self.mask_dilation = config['mask dilation']
            self.wing_size_rank = config["rank wing size"]
            self.do_curriculum_learning = config["do curriculum learning"]
            self.single_time_channel = bool(config["single time channel"])

            # Network configuration
            self.num_blocks = config["number of encoder decoder blocks"]
            self.kernel_size = config["convolution kernel size"]
            self.num_base_filters = config["number of base filters"]
            self.dilation_rate = config["dilation rate"]
            self.dropout = config["dropout ratio"]

            # augmentation configuration
            self.rotation_range = config["rotation range"]
            self.zoom_range = config["zoom range"]
            self.horizontal_flip = bool(config["horizontal flip"])
            self.vertical_flip = bool(config["vertical flip"])
            self.shift = config["xy shift"]
            self.batch_size = config["batch size"] if not self.debug_mode else 1

    def get_config_file(self):
        return self.config

    def get_data_path(self):
        return self.data_path
    
    def get_model_type(self):
        return self.model_type
    
    def get_val_fraction(self):
        return self.val_fraction
    
    def get_num_epochs(self):
        return self.num_epochs

    def get_single_time_channel(self):
        return self.single_time_channel
    
    def get_debug_mode(self):
        return self.debug_mode
    
    def get_mask_dilation(self):
        return self.mask_dilation
    
    def get_mix_with_test(self):
        return self.mix_with_test
    
    def get_base_output_directory(self):
        return self.base_output_directory
    
    def get_augmentation_configuration(self):
        return self.rotation_range,\
            self.zoom_range,\
            self.horizontal_flip,\
            self.vertical_flip,\
            self.shift
    
    def get_network_configuration(self):
        return self.num_base_filters,\
            self.num_blocks,\
            self.kernel_size,\
            self.dilation_rate,\
            self.dropout
    
    def get_resume_training_checkpoint_path(self):
        return self.resume_training_checkpoint_path
    
    def get_resume_training_directory(self):
        return self.resume_training_directory

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
          

def create_run_folders(base_output_directory, run_name, original_config_file):
    """ Creates folders necessary for outputs of vision. """
    run_path = os.path.join(base_output_directory, run_name)
    initial_run_path = run_path
    i = 1
    while os.path.exists(run_path):
        run_path = "%s_%02d" % (initial_run_path, i)
        i += 1
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path)
    os.makedirs(os.path.join(run_path, "weights"))
    os.makedirs(os.path.join(run_path, "viz_pred"))
    os.makedirs(os.path.join(run_path, "histograms"))
    os.makedirs(os.path.join(run_path, "l2_histograms_per_point"))
    print("Created folder:", run_path)
    code_dir_path = os.path.join(run_path, "training code")
    os.makedirs(code_dir_path)
    for file_name in os.listdir('.'):
        if file_name.endswith('.py'):
            full_file_name = os.path.join('.', file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, code_dir_path)
                print(f"Copied {full_file_name} to {code_dir_path}")
    with open(f"{run_path}/configuration.json", 'w') as file:
        json.dump(original_config_file, file, indent=4)
    return run_path

def save_training_code(base_run_directory):
    save_to_directory = os.path.join(base_run_directory, "training code")
    if not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
    for file_name in os.listdir(TRAINING_CODE_DIRECTORY):
        if file_name.endswith('.py'):
            full_file_name = os.path.join(TRAINING_CODE_DIRECTORY, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, save_to_directory)
    for file_name in os.listdir(SBATCH_FILES_DIRECTORY):
        full_file_name = os.path.join(SBATCH_FILES_DIRECTORY, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, save_to_directory)
    print(f"Copied training code to {save_to_directory}")
        