import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sympy import re
matplotlib.use("Agg")
import os
import shutil
import json
import torch
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop
import h5py
import glob
from training_code.Losses import SoftArgmaxLoss, SpatialKLLoss, JSDLoss
import re as regex

TRAINING_CODE_DIRECTORY = "code/training_code"
PREDICTION_CODE_DIRECTORY = "code/prediction_code_lior"
PREDICTION_CONFIGURATIONS_DIRECTORY = "predict_configurations"
SBATCH_FILES_DIRECTORY = "sbatch_files"
VIZ_OUTPUT_DIRECTORY_NAME = "viz_pred"

optimizer_from_string = {
    "Adam": Adam,
    "SGD": SGD,
    "RMSprop": RMSprop,
}

class TrainConfig:
    def __init__(self, config_path):
        with open(config_path) as CF:
            config = json.load(CF)
            # training configuration
            self.config = config
            self.debug_mode = bool(config["debug mode"])
            self.num_epochs = config['epochs']
            self.val_fraction = config['val fraction']
            self.loss_function_as_string = config["loss function"]
            self.learning_rate = config["learning rate"]
            self.optimizer_as_string = config["optimizer"]
            self.optimizer_epsilon = config["optimizer epsilon"]
            self.weight_initialization_method = config["weight initialization method"]
            self.reduce_lr_factor = config["reduce lr factor"]
            self.reduce_lr_patience = config["reduce lr patience"]
            self.reduce_lr_min_delta = config["reduce lr min delta"]
            self.reduce_lr_cooldown = config["reduce lr cooldown"]
            self.reduce_lr_min_lr = config["reduce lr min lr"]
            self.base_output_directory = config["base output directory"]
            self.how_many_visualizations = 1 if self.debug_mode else config.get("how many visualizations", 10)
            self.model_type = config["model type"]
            # Optional short label appended to the auto-generated run folder
            # name (after model type) so variants that share a model type are
            # distinguishable, e.g. "MODEL_PER_CAM_PER_WING_JSD_Jun 30".
            # Defaults to "" -> name is unchanged.
            self.run_tag = config.get("run tag", "")
            self.save_every = config["save every"]
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
            # Normalization for the U-Net backbone: "none" (default, matches the
            # norm-free encoder_atrous/decoder nets), "group", or "batch".
            self.normalization = config.get("normalization", "none")
            # Cross-camera fusion for the multi-view MultiCamNetwork: "concat"
            # (default = original behavior), or the permutation-invariant pools
            # "max" / "mean". Only read by MultiCamNetwork; remove this line and
            # its getter to drop the feature.
            self.camera_fusion = config.get("camera fusion", "concat")
            # How many cameras a multi-view training sample should carry.
            # None (the default) = every camera the dataset has, i.e. exactly
            # the behavior before 3-camera support. Setting it BELOW the
            # dataset's count turns each labelled frame into one sample per
            # camera subset -- see Preprocessor.camera_subsets.
            self.num_cameras = config.get("number of cameras")

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

    def get_run_tag(self):
        return self.run_tag

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
            self.weight_initialization_method,\
            self.dropout

    def get_normalization(self):
        return self.normalization

    def get_camera_fusion(self):
        return self.camera_fusion

    def get_num_cameras(self):
        return self.num_cameras
    
    def get_resume_training_checkpoint_path(self):
        return self.resume_training_checkpoint_path
    
    def get_resume_training_directory(self):
        return self.resume_training_directory
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def configure_loss(self):
        if self.loss_function_as_string == "MSE":
            try:
                reduction = self.config["loss reduction"]
            except KeyError:
                print("No reduction method specified for MSE loss. Using default 'mean'.")
                reduction = "mean"
            if reduction == "mean":
                return MSELoss(reduction="mean")
            elif reduction == "sum":
                return MSELoss(reduction="sum")
        elif self.loss_function_as_string == "KL":
            return SpatialKLLoss()
        elif self.loss_function_as_string == "softargmax":
            return SoftArgmaxLoss()
        elif self.loss_function_as_string == "JSD":
            return JSDLoss()
        else:
            raise ValueError(f"Loss function {self.loss_function_as_string} not recognized.")


# Marker for a model that runs on any camera count (every PER_CAM model, and
# any model.json predating the "num cameras" field).
ANY_NUM_CAMS = "any"


def model_accepts_num_cams(model_config, num_cams):
    """Can this ensemble member run on a movie with `num_cams` cameras?"""
    declared = model_config.get("num cameras", ANY_NUM_CAMS)
    if declared is None or declared == ANY_NUM_CAMS:
        return True
    return int(declared) == int(num_cams)


# Config value meaning "read this off the data instead of trusting me".
AUTO = "auto"


def _explicit_or_auto(value):
    """None when a config field is absent or the literal "auto", else itself."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == AUTO:
        return None
    return value


class PredictConfig:
    def __init__(self, config_path):
        with open(config_path) as CF:
            config = json.load(CF)
            self.input_data_directory = config['data directory']
            self.output_directory = config['output directory']
            # Camera count and calibration are properties of the DATA, not of
            # the config: the old rig recorded 3 cameras and the current one
            # records 4, and every experiment has its own calibration.h5 next
            # to its movies. Leave either out (or set it to "auto") and
            # apply_movie_geometry resolves it per movie, so one config covers
            # both rigs and a stale path can no longer quietly triangulate one
            # experiment's movies against another's calibration.
            # The `_declared_*` pair holds what the config actually said; the
            # public attributes are re-resolved for each movie in
            # apply_movie_geometry. Keeping them apart is what lets one config
            # cover several movies -- if the resolved value stuck, the first
            # movie's camera count would be enforced on every later one.
            self._declared_calibration_path = _explicit_or_auto(config.get('calibration path'))
            self._declared_num_cams = _explicit_or_auto(config.get('number of cameras'))
            self.calibration_data_path = self._declared_calibration_path
            self.num_cams = self._declared_num_cams
            self.wings_detector_path = config['wings detector path']
            self.image_height = config['IMAGE HEIGHT']
            self.image_width = config['IMAGE WIDTH']
            self.mask_increase_initial = config['mask increase initial']
            self.mask_increase_reprojected = config['mask increase reprojected']
            self.is_video = bool(config['is video'])
            self.batch_size = config['batch size']
            # Optional: name of the shared parent directory grouping all
            # movies of a batch. Set by predict_array.sh from SLURM_JOB_NAME
            # so every task in an array lands in one parent. Falls back to
            # the data-directory basename when absent.
            self.general_run_name = config.get('general run name') or None
            # Optional: append per-movie predict timing rows to this CSV.
            # Set by predict_array.sh from pipeline.sh's TIMINGS_PATH.
            self.pipeline_timings_path = config.get('pipeline timings path') or None
            # Optional: cap the largest model-subset the ensemble selector
            # searches over. The search is ~2^M in the number of members, so a
            # cap (e.g. 3) keeps a large ensemble tractable. None => no cap.
            self.max_ensemble_models = config.get('max ensemble models')

            # Preferred: auto-discover the ensemble from a prediction_models/
            # directory (each subfolder is a self-contained model). Falls back
            # to the legacy config-bank + specified-configs pair when absent.
            pred_models_dir = config.get('prediction models directory')
            if pred_models_dir:
                self.model_config_list = self.load_configurations_from_models_dir(pred_models_dir)
            else:
                config_bank_path = config['config bank path']
                specified_configs_path = config['specified configs path']
                self.model_config_list = self.load_configurations_from_bank(config_bank_path, specified_configs_path)
            self.tuned_configration = False

    def load_configurations_from_models_dir(self, pred_models_dir):
        """Build the ensemble member list by scanning a prediction_models/ dir.

        Every subfolder that holds ``best_model.pt`` + ``model.json`` and is not
        disabled becomes a member, in deterministic (sorted) order. Produces the
        same dict shape as load_configurations_from_bank so the rest of the
        pipeline is unchanged, plus a ``name`` (the folder name) used for tidy
        per-member output directories and the model-selection report."""
        model_config_list = []
        for model_dir in sorted(glob.glob(os.path.join(pred_models_dir, "*"))):
            if not os.path.isdir(model_dir):
                continue
            weights_path = os.path.join(model_dir, "best_model.pt")
            meta_path = os.path.join(model_dir, "model.json")
            if not (os.path.isfile(weights_path) and os.path.isfile(meta_path)):
                continue
            with open(meta_path) as MF:
                meta = json.load(MF)
            if not meta.get("enabled", True):
                continue
            model_type = meta["model type"]
            model_config_list.append({
                "name": os.path.basename(model_dir.rstrip(os.sep)),
                "wings pose estimation model path": weights_path,
                "wings pose estimation model path second pass": weights_path,
                "model type": model_type,
                "model type second pass": meta.get("model type second pass", model_type),
                "predict again 3D consistency": meta.get("predict again 3D consistency", 0),
                "use reprojected masks": meta.get("use reprojected masks", 0),
                # How many cameras this model's weights are shaped for. An
                # ALL_CAMS model fuses a fixed number of camera streams and
                # simply cannot run on a movie with a different count; a
                # PER_CAM model sees one camera at a time and works with any.
                # Absent => "any", so every model.json written before this
                # existed keeps working.
                "num cameras": meta.get("num cameras", ANY_NUM_CAMS),
            })
        if not model_config_list:
            raise ValueError(f"No enabled prediction models found in {pred_models_dir}")
        return model_config_list

    def load_configurations_from_bank(self, config_bank_path, specified_configs_path):
        with open(config_bank_path) as CBF:
            config_bank = json.load(CBF)
        with open(specified_configs_path) as SCF:
            specified_configs = json.load(SCF)["specified configurations"]
        model_config_list = []
        for config_name in specified_configs:
            if config_name in config_bank:
                model_config_list.append(config_bank[config_name])
            else:
                raise ValueError(f"Config name {config_name} not found in config bank.")
        return model_config_list

    def get_input_data_directory(self):
        return self.input_data_directory
    
    def get_output_directory(self):
        return self.output_directory

    def get_general_run_name(self):
        return self.general_run_name

    def get_pipeline_timings_path(self):
        return self.pipeline_timings_path

    def get_max_ensemble_models(self):
        return self.max_ensemble_models

    def get_calibration_path(self):
        return self.calibration_data_path
    
    def get_model_type(self):
        if not self.tuned_configration:
            raise ValueError("Predict_config not finished configuring. Call finish_configuring() first.")
        return self.model_type
    
    def get_wings_detector_path(self):
        return self.wings_detector_path
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_model_config_list(self, num_cams=None):
        """The ensemble members, optionally restricted to those that can run
        on a `num_cams`-camera movie. Called per movie, since the count is a
        property of the data and one config may cover several movies."""
        if num_cams is None:
            return self.model_config_list
        return [m for m in self.model_config_list
                if model_accepts_num_cams(m, num_cams)]

    @staticmethod
    def resolve_calibration_path(movie_path):
        """The calibration.h5 that belongs to this movie, nearest first.

        The build step writes one calibration.h5 per experiment directory in
        multi-movie mode, and into the movie directory itself for a
        single-movie build -- so check both, and fail loudly rather than fall
        back to some other experiment's file."""
        movie_dir = os.path.dirname(os.path.abspath(movie_path))
        for cand in (os.path.join(movie_dir, "calibration.h5"),
                     os.path.join(os.path.dirname(movie_dir), "calibration.h5")):
            if os.path.isfile(cand):
                return cand
        raise SystemExit(
            f"no calibration.h5 in {movie_dir} or its parent, and the config "
            f"gives no explicit 'calibration path'. Run the build step first.")

    def apply_movie_geometry(self, movie_path, num_cams):
        """Resolve the config fields that belong to the movie, not the config.

        `num_cams` is read off the box's own cropzone, so a 3-camera movie is
        recognised as such whatever the config says, and the ensemble members
        that cannot run on it are dropped by describe_model_selection. An
        explicit config value still wins, but a value that CONTRADICTS the box
        is fatal: every downstream tensor shape is derived from this number,
        so a silent mismatch would produce garbage rather than an error.
        """
        declared = self._declared_num_cams
        if declared is None:
            if num_cams is None:
                raise SystemExit(
                    f"could not read the camera count from {movie_path} and "
                    f"the config does not set 'number of cameras'.")
            self.num_cams = int(num_cams)
        elif num_cams is not None and int(declared) != int(num_cams):
            raise SystemExit(
                f"config says 'number of cameras': {declared} but "
                f"{os.path.basename(movie_path)} has {num_cams} cameras in "
                f"its box. Set the field to \"auto\", or point the config at "
                f"the right experiment.")
        else:
            self.num_cams = int(declared)
        self.calibration_data_path = (
            self._declared_calibration_path
            if self._declared_calibration_path is not None
            else self.resolve_calibration_path(movie_path))
        return self.num_cams, self.calibration_data_path

    def describe_model_selection(self, num_cams):
        """(kept, [(name, why-skipped)]) for logging which members will run."""
        kept, skipped = [], []
        for m in self.model_config_list:
            if model_accepts_num_cams(m, num_cams):
                kept.append(m)
            else:
                skipped.append((m.get("name", m["model type"]),
                                f"needs {m['num cameras']} cameras"))
        return kept, skipped
    
    def tune_configuration(self, config_as_dict, movie_path, specific_output_directory):
        self.wings_pose_estimation_model_path = config_as_dict["wings pose estimation model path"]
        self.wings_pose_estimation_model_path_second_pass = config_as_dict["wings pose estimation model path second pass"]
        self.model_type = config_as_dict["model type"]
        self.model_type_second_pass = config_as_dict["model type second pass"]
        self.predict_again_3D_consistency = config_as_dict["predict again 3D consistency"]
        self.use_reprojected_masks = bool(config_as_dict["use reprojected masks"])
        self.movie_path = movie_path
        self.specific_output_directory = specific_output_directory
        if not self.tuned_configration:
            self.tuned_configration = True

    def get_predictor_data(self):
        if not self.tuned_configration:
            raise ValueError("Predict_config not finished configuring. Call finish_configuring() first.")
        return \
        self.model_type, \
        self.model_type_second_pass, \
        self.movie_path, \
        self.wings_detector_path, \
        self.wings_pose_estimation_model_path, \
        self.wings_pose_estimation_model_path_second_pass, \
        self.specific_output_directory, \
        self.is_video, \
        self.batch_size, \
        self.num_cams, \
        self.mask_increase_initial, \
        self.mask_increase_reprojected, \
        self.predict_again_3D_consistency, \
        self.use_reprojected_masks

    def get_triangulator_data(self):
        return self.calibration_data_path, \
                self.image_height, \
                self.image_width

    def get_full_config_as_dict(self):
        if not self.tuned_configration:
            raise ValueError("Predict_config not finished configuring. Call finish_configuring() first.")
        return {
            "input data directory": self.input_data_directory,
            "movie path": self.movie_path,
            "output directory": self.output_directory,
            "specific output directory": self.specific_output_directory,
            "calibration path": self.calibration_data_path,
            "wings detector path": self.wings_detector_path,
            "IMAGE HEIGHT": self.image_height,
            "IMAGE WIDTH": self.image_width,
            "number of cameras": self.num_cams,
            "mask increase initial": self.mask_increase_initial,
            "mask increase reprojected": self.mask_increase_reprojected,
            "is video": self.is_video,
            "batch size": self.batch_size,
            "wings pose estimation model path": self.wings_pose_estimation_model_path,
            "wings pose estimation model path second pass": self.wings_pose_estimation_model_path_second_pass,
            "model type": self.model_type,
            "model type second pass": self.model_type_second_pass,
            "predict again 3D consistency": self.predict_again_3D_consistency,
            "use reprojected masks": self.use_reprojected_masks
        }

    def save_config_as_json(self, save_directory, filename="specific_configuration.json"):
        if not self.tuned_configration:
            raise ValueError("Predict_config not finished configuring. Call finish_configuring() first.")
        config_dict = self.get_full_config_as_dict()
        file_path = os.path.join(save_directory, filename)
        with open(file_path, 'w') as file:
            json.dump(config_dict, file, indent=4)
        print(f"Saved used configuration to {file_path}")


def tf_format_find_peaks(x):
        '''
        find peaks in confidence maps.
        Args:
            x: np.array of shape [batch, height, width, channels]
        Returns:
            pred: np.array of shape [batch, 3, channels], for each channel:
                  [0,:] = x-coords (cols),
                  [1,:] = y-coords (rows),
                  [2,:] = peak values.
        '''
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

def torch_find_peaks(x):
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
    label_batch = label

    # Find peaks -> shape [B,3,C]
    peaks = torch_find_peaks(label_batch)  # (x, y, val)
    coords = peaks[:, :2, :].transpose(0, 2, 1)[0]  # shape (num_points, 2)

    _, _, H, W = sample.shape
    num_points = coords.shape[0]

    # Use the 2nd channel (index 1) of sample as background
    frame = sample[0, 1, :, :]

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
          

def create_train_run_folders(base_output_directory, run_name, original_config_file):
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
    code_directory = os.path.dirname(TRAINING_CODE_DIRECTORY)
    for file_name in os.listdir(code_directory):
        if file_name.endswith('.py'):
            full_file_name = os.path.join(code_directory, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, save_to_directory)
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
        
def readfile(path):
    with (h5py.File(path, "r") as f):
        keys = list(f.keys())
        for key in keys:
            value = f[key]
            print(f'key:{key}, value:{value}')

def predict_3D_points_all_pairs(base_path):
    all_points_file_list = []
    points_3D_file_list = []
    dir_path = os.path.join(base_path)
    dirs = glob.glob(os.path.join(dir_path, "*"))
    for dir in dirs:
        if os.path.isdir(dir):
            all_points_file = os.path.join(dir, "points_3D_all.npy")
            points_3D_file = os.path.join(dir, "points_3D.npy")
            if os.path.isfile(all_points_file):
                all_points_file_list.append(all_points_file)
            if os.path.isfile(points_3D_file):
                points_3D_file_list.append(points_3D_file)
    all_points_arrays = [np.load(array_path) for array_path in all_points_file_list]
    points_3D_arrays = [np.load(array_path)[:, :, np.newaxis, :] for array_path in points_3D_file_list]
    big_array_all_points = np.concatenate(all_points_arrays, axis=2)
    return big_array_all_points, all_points_arrays

def add_nan_frames(original_array, N):
    nan_frames = np.full((N,) + original_array.shape[1:], np.nan)
    new_array = np.concatenate((nan_frames, original_array), axis=0)
    return new_array

def find_starting_frame(readme_file):
    start_pattern = re.compile(r'start:\s*(\d+)')
    with open(readme_file, 'r') as file:
        for line in file:
            match = start_pattern.search(line)
            if match:
                start_number = match.group(1)
                return start_number

def get_start_frame(movie_dir_path):
    start_frame = 0
    for filename in os.listdir(movie_dir_path):
        if filename.startswith("README_mov"):
            readme_file = os.path.join(movie_dir_path, filename)
            start_frame = find_starting_frame(readme_file)
    return start_frame

def draw_sample_with_points(sample_image, predicted_points, gt_points, save_file_path):
    """
    Saves a visualization of ground truth and predicted landmarks on a sample image.
    
    GT: Hollow circles (color coded by index).
    Pred: 'x' markers (color coded by index).
    """
    try:
        # 1. Create a figure and axes
        fig, ax = plt.subplots()

        # 2. Handle data type range for imshow
        img_to_show = sample_image
        v_min, v_max = None, None
        
        if img_to_show.dtype in (np.float32, np.float64):
            if img_to_show.max() > 1.0:
                v_min, v_max = 0, 255

        # 3. Display the grayscale image
        ax.imshow(img_to_show, cmap='gray', vmin=v_min, vmax=v_max)

        # GENERATE COLORS
        # Create a unique color for each point index.
        # We use the 'rainbow' colormap to ensure distinction.
        num_points = len(gt_points)
        colors = cm.rainbow(np.linspace(0, 1, num_points))

        # 4. Draw Ground Truth points 
        # Requirement: Empty circle, very narrow line, color-coded.
        ax.scatter(
            gt_points[:, 0], 
            gt_points[:, 1], 
            edgecolors=colors,    # Sets the outline color to our generated array
            facecolors='none',    # Makes the center of the circle transparent/empty
            marker='o', 
            s=30,                 # Size: Increased slightly so the "hollow" part is visible
            linewidths=0.5,       # "Very narrow" line width
            label='Ground Truth'
        )

        ax.scatter(
            gt_points[:, 0], 
            gt_points[:, 1], 
            c=colors,             # Color the dot (filled)
            marker='o',           # Circle marker
            s=2,                  # Very small size for the "dot"
            linewidths=0,         # No border on the dot
            label='Ground Truth (Center)'
        )

        # 5. Draw Predicted points
        # Requirement: 'x' marker, narrower line, matching color to GT.
        ax.scatter(
            predicted_points[:, 0], 
            predicted_points[:, 1], 
            c=colors,             # Sets the line color of the 'x'
            marker='x', 
            s=30,                 # Size matches GT for consistency
            linewidths=0.5,       # "Narrower" line width
            label='Prediction'
        )

        # Note: The text numbering loops have been removed as requested.

        # 6. Clean up the plot
        ax.axis('off')
        
        # 7. Save the final image
        fig.savefig(
            save_file_path, 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=150 
        )
        
        # 8. Close the figure
        plt.close(fig)

    except Exception as e:
        print(f"Error saving image to {save_file_path} using Matplotlib: {e}")

def show_pred(net, sample_image, gt_confmaps, epoch_num, save_directory):
    net.eval()
    x_batch = sample_image[None, ...]
    try:
        device = next(net.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    x_tensor = torch.from_numpy(x_batch).float().to(device)
    with torch.no_grad():
        predicted_confmaps = net(x_tensor)
        predicted_confmaps = predicted_confmaps.cpu().numpy()

    predicted_peaks = torch_find_peaks(predicted_confmaps)[0,:2,:]
    gt_peaks = torch_find_peaks(gt_confmaps[None, ...])[0,:2,:]
    save_path = os.path.join(save_directory, f"epoch_{epoch_num}.png")
    draw_sample_with_points(
        sample_image=np.squeeze(sample_image)[1,...],
        predicted_points=predicted_peaks.T,
        gt_points=gt_peaks.T,
        save_file_path=save_path
    )

def show_pred_multiple_cameras(net, sample, gt_confmaps, epoch_num, save_directory, num_cameras, num_points):
    net.eval()
    x_batch = sample[None, ...]
    try:
        device = next(net.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    x_tensor = torch.from_numpy(x_batch).float().to(device)
    with torch.no_grad():
        predicted_confmaps = net(x_tensor)
        predicted_confmaps = predicted_confmaps.cpu().numpy()

    predicted_peaks = torch_find_peaks(predicted_confmaps)[0,:2,:]
    gt_peaks = torch_find_peaks(gt_confmaps[None, ...])[0,:2,:]
    channels_per_cam = sample.shape[0] // num_cameras
    for i in range(num_cameras):
        current_gt_peaks = gt_peaks[:, i*num_points:(i+1)*num_points]
        current_predicted_peaks = predicted_peaks[:, i*num_points:(i+1)*num_points]
        # the "present" time channel of camera i
        current_image = sample[1 + i*channels_per_cam]
        save_path = os.path.join(save_directory, f"epoch_{epoch_num}_cam{i+1}.png")
        draw_sample_with_points(
            sample_image=np.squeeze(current_image),
            predicted_points=current_predicted_peaks.T,
            gt_points=current_gt_peaks.T,
            save_file_path=save_path
        )

def find_flip_in_files(movie_dir_path):
    # Word to search for
    word_to_search = "flip"

    # Regular expression pattern to match filenames like README_mov{some number}.txt
    pattern = regex.compile(r"README_mov\d+\.txt")

    try:
        # List all files in the directory
        for filename in os.listdir(movie_dir_path):
            # Check if the filename matches the pattern
            if pattern.match(filename):
                file_path = os.path.join(movie_dir_path, filename)
                # Open the file and search for the word
                with open(file_path, 'r') as file:
                    for line in file:
                        if word_to_search in line:
                            return True
        return False
    except FileNotFoundError:
        # If the directory does not exist, return False
        return False


# pitch_angle and pitch_dot are nose-DOWN positive -- the right-hand rule about
# y_body, which points left -- so they share a sign with omega_body about y_body.
# An analysis h5 records this in a text dataset named PITCH_CONVENTION_KEY. A file
# written before the convention was fixed has no such dataset and holds the
# opposite, nose-up sign; multiply what is read by pitch_read_sign to undo that.
PITCH_CONVENTION = "nose_down_positive"
PITCH_CONVENTION_KEY = "pitch_convention"
PITCH_SIGNED_KEYS = ("pitch_angle", "pitch_dot", "pitch_dot_dot")


def pitch_read_sign(h5, key):
    """-1 if `key` is a pitch dataset in a file still on the old nose-up sign, else +1."""
    if key in PITCH_SIGNED_KEYS and PITCH_CONVENTION_KEY not in h5:
        return -1
    return 1


# Written next to each movie by process_experiment's prescan; see
# process_experiment.write_cam_validity_sidecar.
# Declared by process_experiment.py --perturbation, and read back at predict
# time. It sits next to calibration.h5 so the declaration follows the data
# rather than a config that is shared across experiments; its mere presence is
# what marks an experiment as a perturbation experiment.
PERTURBATION_FILE = "perturbation.json"

# Per-frame perturbation state. UNKNOWN is a first-class value, not a failure:
# an experiment whose log recorded the onset but never the duration genuinely
# cannot say whether a frame past the onset is still during the perturbation or
# after it, and labelling those frames "not after" would assert something the
# record does not support.
PERT_UNKNOWN = -1
PERT_BEFORE = 0
PERT_DURING = 1
PERT_AFTER = 2
# A movie the declaration explicitly marks as UNPERTURBED. Distinct from
# PERT_UNKNOWN: "we know there was no perturbation" is a positive fact and a
# usable experimental control, while "unknown" is an absence of knowledge.
PERT_CONTROL = 3
PERT_STATE_NAMES = {PERT_UNKNOWN: "unknown", PERT_BEFORE: "before",
                    PERT_DURING: "during", PERT_AFTER: "after",
                    PERT_CONTROL: "control"}

# The three per-movie statuses a declaration can assign.
PERT_STATUSES = ("perturbed", "control", "unknown")

# Applied when a movie is known to be perturbed but the log never recorded how
# long the pulse lasted. Source: Noam Tsory's MSc thesis, section 4.1.3, which
# describes the rig as firing "the magnetic perturbation using pre-set duration
# (7.5 ms)" -- a property of the apparatus, not of any one experiment.
# It is NEVER applied silently: process_experiment writes the number and its
# provenance into perturbation.json, load_perturbation reports
# duration_source == "assumed", and every product carries that word through.
PERT_DEFAULT_DURATION_MS = 7.5

# LIGHTING -- a second stimulus axis, independent of the magnetic pulse. An
# experiment can be lit throughout, dark throughout, or have its white light
# switched OFF during the recording; that switch is a visual perturbation in its
# own right (Tsory thesis s.5.2), so a movie holding one is a two-stimulus movie
# even when its coil pulse comes later. Declared in the same perturbation.json,
# in a "lighting" block that a per-movie entry can override exactly like the
# pulse. The word "dark" in a directory name can mean either of the two dark
# regimes, which is precisely why the regime has to be stated rather than read
# off a name.
LIGHTING_REGIMES = ("constant_light", "constant_dark", "darkening", "unknown")
_LIGHTING_ALIASES = {
    "light": "constant_light", "lit": "constant_light",
    "constant light": "constant_light",
    "dark": "constant_dark", "constant_darkness": "constant_dark",
    "constant dark": "constant_dark", "constant darkness": "constant_dark",
    "darkening_at_trigger": "darkening", "dark_at_trigger": "darkening",
    "light_off": "darkening",
}
# Per-frame lighting state, the lighting counterpart of PERT_*.
LIGHT_UNKNOWN = -1
LIGHT_LIT = 0
LIGHT_DARK = 1
LIGHT_STATE_NAMES = {LIGHT_UNKNOWN: "unknown", LIGHT_LIT: "lit", LIGHT_DARK: "dark"}
# How long the light stays off once the trigger switches it. Tsory thesis
# s.4.1.3: the trigger turns "off the ambient white light ... and turn on again
# after 1s". A property of the rig's Arduino program, like the 7.5 ms pulse.
LIGHT_DEFAULT_RELIGHT_MS = 1000.0


def lighting_is_darkening(pert):
    """True when the light is switched OFF during this movie's timeline."""
    return (pert is not None and pert.get("lighting_regime") == "darkening"
            and pert.get("light_off_frame") is not None)


def pert_is_perturbed(pert):
    """True when this movie actually carries a perturbation window.

    The one predicate every consumer should branch on. `load_perturbation`
    returns a dict for control and unknown movies too -- that is how a mixed
    experiment declares which of its movies are controls -- so `pert is not
    None` no longer means "perturbed"."""
    return pert is not None and pert.get("status") == "perturbed"


def resolve_perturbation_path(movie_path):
    """perturbation.json for a movie, nearest first, or None when absent.

    Same search order as PredictConfig.resolve_calibration_path: the movie's
    own directory (single-movie builds) then its parent (the experiment dir,
    where a multi-movie build puts calibration.h5)."""
    movie_dir = os.path.dirname(os.path.abspath(movie_path))
    for cand in (os.path.join(movie_dir, PERTURBATION_FILE),
                 os.path.join(os.path.dirname(movie_dir), PERTURBATION_FILE)):
        if os.path.isfile(cand):
            return cand
    return None


def load_perturbation(movie_path, frame_rate=None):
    """What the experiment declares about THIS movie, or None if it declares nothing.

    Returns None only when no perturbation.json applies (absent, or unreadable).
    Whenever a declaration is found a dict comes back -- including for a movie
    the declaration marks as a CONTROL or as UNKNOWN. That is what lets one
    experiment hold both perturbed and unperturbed movies: `pert is not None`
    means "declared", and `pert_is_perturbed(pert)` means "perturbed".

    A per-movie entry in `movies` (keyed by the movie's directory basename)
    overrides the experiment-level block key by key, because the parts of one
    experiment can differ -- ex241220's dark parts ran 7.5 ms and 12 ms off the
    same declaration.

    The duration is stored in MILLISECONDS and converted here using the movie's
    own frame_rate, so an experiment recorded at more than one frame rate cannot
    silently acquire the wrong window. `end_known` says whether a duration is
    known at all; `end_frame` is separately None when it could not be located in
    frames (no frame rate) -- those are different failures and callers that
    place the boundary must branch on `end_frame is not None`.

    NOTE: a `"usable": false` flag in a movies entry means the movie was
    excluded from the build. It is NOT a status and is never mapped to one.

    The LIGHTING is resolved from the same file, from a "lighting" block (and a
    per-movie "lighting" override), into the flat keys `lighting_declared`,
    `lighting_regime`, `light_off_frame`, `light_on_frame`, `relight_after_ms`,
    `lighting_note` and `lighting_evidence` -- see _resolve_lighting. They are
    present on every returned dict, control and unknown movies included: the
    light is a property of the session, not of whether this fly had a magnet.
    """
    path = resolve_perturbation_path(movie_path)
    if path is None:
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {path}: {e}; treating the experiment as "
              f"un-perturbed", flush=True)
        return None

    base = doc.get("perturbation") or {}
    movie_key = os.path.basename(os.path.dirname(os.path.abspath(movie_path)))
    over = (doc.get("movies") or {}).get(movie_key) or {}

    def pick(*keys, default=None):
        for src in (over, base):
            for k in keys:
                if src.get(k) is not None:
                    return src[k]
        return default

    status = str(pick("status", default="perturbed")).strip().lower()
    if status not in PERT_STATUSES:
        print(f"{path}: unrecognised perturbation status {status!r} for "
              f"{movie_key}; treating as 'unknown'", flush=True)
        status = "unknown"

    # A type that was never really stated must not be presented as one.
    raw_type = pick("type", default=None)
    type_known = str(raw_type).strip().lower() not in (
        "", "none", "unspecified", "unknown") if raw_type is not None else False

    onset = duration_ms = None
    duration_source = "n/a"
    if status == "perturbed":
        onset = int(pick("onset_trigger_frame", "onset_frame", default=0))
        declared = pick("duration_ms")
        if declared is not None:
            duration_ms = float(declared)
            # A declaration may state its own provenance -- a duration copied
            # from a rig-wide default is NOT a measurement of this experiment,
            # and only the file that wrote it knows which it is. Inferring
            # "recorded" from mere presence would launder an assumption into a
            # fact, so an explicit duration_source always wins.
            stated = pick("duration_source")
            duration_source = (str(stated).strip().lower()
                               if str(stated).strip().lower() in
                               ("recorded", "assumed", "unrecorded")
                               else "recorded")
        elif bool(pick("assume_duration", default=True)):
            duration_ms = float(pick("duration_assumed_ms",
                                     default=PERT_DEFAULT_DURATION_MS))
            duration_source = "assumed"
        else:
            duration_source = "unrecorded"

    end_frame = None
    if duration_ms is not None and onset is not None and frame_rate:
        end_frame = onset + int(round(float(duration_ms) * float(frame_rate) / 1000.0))

    return {
        "status": status,
        "type": str(raw_type) if raw_type is not None else "unknown",
        "type_known": type_known,
        "onset_frame": onset,
        "duration_ms": duration_ms,
        "duration_source": duration_source,
        "duration_assumed_ms": (duration_ms if duration_source == "assumed"
                                else None),
        "duration_note": pick("duration_status", default=None),
        "end_frame": end_frame,
        # "is a duration known at all", independent of whether it could be
        # located in frames -- see the docstring.
        "end_known": duration_ms is not None,
        "frame_rate": float(frame_rate) if frame_rate else None,
        "movie_key": movie_key,
        "source": path,
        **_resolve_lighting(doc, over, frame_rate, path, movie_key),
    }


def _resolve_lighting(doc, over, frame_rate, path, movie_key):
    """The lighting half of a declaration, as flat keys for load_perturbation's dict.

    movies[<dir>]["lighting"] overrides the experiment-level "lighting" block
    key by key. A file with no lighting block anywhere says nothing about the
    light: that comes back as `lighting_declared` False with regime "unknown",
    never as "lit".

    For a darkening, `light_off_frame` is trigger-relative and `light_on_frame`
    is where the light comes back (`relight_after_ms` later, located with this
    movie's own frame rate -- None when there is no frame rate).
    """
    base = doc.get("lighting") or {}
    mine = (over or {}).get("lighting") or {}

    def pick(key, default=None):
        for src in (mine, base):
            if src.get(key) is not None:
                return src[key]
        return default

    raw = str(pick("regime", default="unknown")).strip().lower()
    regime = _LIGHTING_ALIASES.get(raw, raw)
    if regime not in LIGHTING_REGIMES:
        print(f"{path}: unrecognised lighting regime {raw!r} for {movie_key}; "
              f"treating as 'unknown'", flush=True)
        regime = "unknown"

    off = on = relight_ms = None
    if regime == "darkening":
        off_raw = pick("light_off_trigger_frame")
        off = int(off_raw) if off_raw is not None else None
        relight_ms = float(pick("relight_after_ms", default=LIGHT_DEFAULT_RELIGHT_MS))
        if off is not None and frame_rate:
            on = off + int(round(relight_ms * float(frame_rate) / 1000.0))
    return {
        "lighting_declared": bool(base or mine),
        "lighting_regime": regime,
        "light_off_frame": off,
        "light_on_frame": on,
        "relight_after_ms": relight_ms,
        "lighting_note": pick("note", default=None),
        "lighting_evidence": pick("evidence", default=None),
    }


def lighting_frame_labels(frame_numbers, pert, trigger_relative=True):
    """(state, off_index, on_index): was each trigger-relative frame LIT or DARK.

    Constant regimes need no clock -- every frame is lit, or every frame is
    dark, whatever the numbering -- so they are labelled even without the
    trigger. A darkening can only be located in trigger-relative frames; without
    them, or without a light-off frame, every frame is LIGHT_UNKNOWN.

    DARK is the half-open interval [light_off, light_on), the same convention as
    DURING for the pulse. A relight that could not be located in frames leaves
    the frames from the light-off onward UNKNOWN rather than asserting that the
    light stayed off.

    `off_index` / `on_index` are positions within `frame_numbers`, or -1 when
    that boundary lies outside it.
    """
    f = np.asarray(frame_numbers)
    regime = (pert or {}).get("lighting_regime") or "unknown"
    if regime == "constant_light":
        return np.full(f.shape, LIGHT_LIT, dtype=np.int8), -1, -1
    if regime == "constant_dark":
        return np.full(f.shape, LIGHT_DARK, dtype=np.int8), -1, -1
    off = (pert or {}).get("light_off_frame")
    if regime != "darkening" or off is None or not trigger_relative:
        return np.full(f.shape, LIGHT_UNKNOWN, dtype=np.int8), -1, -1
    on = pert.get("light_on_frame")
    state = np.full(f.shape, LIGHT_UNKNOWN, dtype=np.int8)
    state[f < off] = LIGHT_LIT
    if on is not None:
        state[(f >= off) & (f < on)] = LIGHT_DARK
        state[f >= on] = LIGHT_LIT

    def index_of(target):
        if target is None:
            return -1
        hit = np.nonzero(f == target)[0]
        return int(hit[0]) if len(hit) else -1

    return state, index_of(off), index_of(on)


def declaration_summary(pert):
    """The resolved declaration as two plain dicts, for provenance files."""
    if pert is None:
        return {"perturbation": None, "lighting": None}
    return {
        "perturbation": {k: pert.get(k) for k in (
            "status", "type", "type_known", "onset_frame", "end_frame",
            "duration_ms", "duration_source", "movie_key", "source")},
        "lighting": {
            "declared": pert.get("lighting_declared"),
            "regime": pert.get("lighting_regime"),
            "light_off_frame": pert.get("light_off_frame"),
            "light_on_frame": pert.get("light_on_frame"),
            "relight_after_ms": pert.get("relight_after_ms"),
            "note": pert.get("lighting_note"),
        },
    }


def stamp_declaration(run_dir, pert):
    """Add the resolved pulse + lighting declaration to a movie's source.json.

    source.json says where a movie came from; with this it also says what the
    movie WAS, so the one small file travelling with the outputs is enough to
    tell a darkening movie from a constant-dark one."""
    path = os.path.join(run_dir, "source.json")
    doc = {}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            doc = {}
    doc.update(declaration_summary(pert))
    try:
        with open(path, "w") as f:
            json.dump(doc, f, indent=4, default=str)
    except OSError as e:
        print(f"could not update {path}: {e}", flush=True)


def perturbation_frame_labels(frame_numbers, pert, trigger_relative=True):
    """(state, start_index, end_index) for a sequence of trigger-relative frames.

    `state` is one int8 per frame. For a perturbed movie: PERT_BEFORE /
    PERT_DURING / PERT_AFTER, or PERT_UNKNOWN from the onset onward when no
    duration is known. Frames BEFORE the onset stay exactly labelled either way
    -- the onset is known even when the duration is not.

    DURING is the half-open interval [onset, end): `end_frame` is the first
    frame that is AFTER the perturbation. Every product must agree on this.

    A movie declared CONTROL is labelled PERT_CONTROL throughout, and one
    declared UNKNOWN is PERT_UNKNOWN throughout -- neither has a window, so
    neither has boundaries to index.

    `trigger_relative=False` says the caller could not establish the trigger, so
    these frame numbers are box indices and the window cannot be located in
    them. Everything is then PERT_UNKNOWN, which is the honest answer -- the
    alternative is labelling against a fabricated origin.

    `start_index` / `end_index` are positions within `frame_numbers`, or -1
    when that boundary lies outside the built range. That is a normal outcome,
    not an error: the prescan picks its range from fly visibility, so a movie
    can legitimately begin after the perturbation started.
    """
    f = np.asarray(frame_numbers)
    if pert.get("status") == "control":
        return np.full(f.shape, PERT_CONTROL, dtype=np.int8), -1, -1
    if (not trigger_relative or pert.get("status") != "perturbed"
            or pert.get("onset_frame") is None):
        return np.full(f.shape, PERT_UNKNOWN, dtype=np.int8), -1, -1

    onset = pert["onset_frame"]
    end = pert.get("end_frame")
    state = np.full(f.shape, PERT_UNKNOWN, dtype=np.int8)
    state[f < onset] = PERT_BEFORE
    if end is not None:
        state[(f >= onset) & (f < end)] = PERT_DURING
        state[f >= end] = PERT_AFTER

    def index_of(target):
        hit = np.nonzero(f == target)[0]
        return int(hit[0]) if len(hit) else -1

    return (state, index_of(onset),
            index_of(end) if end is not None else -1)


CAM_VALIDITY_SIDECAR = "prescan_cam_validity.npz"


def load_cam_validity(box_h5_path, num_frames=None, num_cams=None):
    """Which cameras saw the WHOLE fly at each frame of a movie's box.

    Returns a (num_frames, num_cams) bool array, or None when there is no
    sidecar -- which is the case for every movie built before cam-validity
    existed, and the reason callers must treat None as "trust every camera".

    The prescan admits a frame when at least --prescan-min-cams-in-frame
    cameras see the fly whole, so the remaining cameras may be showing a
    TRUNCATED fly. Their 2D detections there are meaningless, and every
    triangulated camera pair that includes such a camera inherits that. This
    array is what lets the prediction stage drop those pairs per frame.

    Shape is validated against the box rather than trusted: a mask that is off
    by even one frame silently blames the wrong cameras, which is worse than
    no mask at all.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(box_h5_path)),
                        CAM_VALIDITY_SIDECAR)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            mask = np.asarray(z["in_frame"], dtype=bool)
    except Exception as e:
        print(f"could not read {path}: {e}; treating every camera as valid",
              flush=True)
        return None
    if mask.ndim != 2:
        print(f"{path}: expected a 2D (frames, cams) mask, got {mask.shape}; "
              f"treating every camera as valid", flush=True)
        return None
    if num_frames is not None and mask.shape[0] != num_frames:
        print(f"{path}: covers {mask.shape[0]} frames but the box has "
              f"{num_frames}; treating every camera as valid", flush=True)
        return None
    if num_cams is not None and mask.shape[1] != num_cams:
        print(f"{path}: covers {mask.shape[1]} cams but the box has "
              f"{num_cams}; treating every camera as valid", flush=True)
        return None
    return mask


def get_trigger_frame_info(box_h5_path):
    """Map a movie's box-frame index to the lab's trigger-relative frame number.

    The high-speed cameras record around a hardware trigger. Each source
    ``*_sparse.mat`` stores ``metaData.startFrame`` — the trigger-relative index
    of the movie's FIRST raw frame (negative => recording began before the
    trigger) — and ``metaData.frameRate`` (Hz). The MATLAB builder then crops the
    raw movie to the 1-based inclusive window ``[start_ind, end_ind]``, encoding
    ``start_ind`` in the h5 filename (``mov_<n>_<start>_<end>_ds_..``). So box
    frame ``k`` (0-based) is raw 1-based frame ``start_ind + k``, whose
    trigger-relative number is ``startFrame + (start_ind - 1) + k``.

    Returns ``(trigger_offset, frame_rate)`` where the trigger-relative number of
    box frame ``k`` is ``trigger_offset + k`` and its time is
    ``(trigger_offset + k) * 1000 / frame_rate`` ms (frame 0 == the trigger).
    Returns ``(None, None)`` if the filename or sparse .mat can't be read.
    """
    try:
        fname = os.path.basename(box_h5_path)
        match = regex.match(r"mov_\d+_(\d+)_(\d+)_ds_", fname)
        start_ind = int(match.group(1)) if match else 1
        movie_dir = os.path.dirname(box_h5_path)
        sparse_mats = sorted(glob.glob(os.path.join(movie_dir, "*_sparse.mat")))
        if not sparse_mats:
            print(f"get_trigger_frame_info: no *_sparse.mat next to {box_h5_path}",
                  flush=True)
            return None, None
        # v7.3 .mat is HDF5; read the two scalars from metaData.
        with h5py.File(sparse_mats[0], "r") as f:
            start_frame = float(np.array(f["metaData"]["startFrame"][()]).squeeze())
            frame_rate = float(np.array(f["metaData"]["frameRate"][()]).squeeze())
        trigger_offset = int(round(start_frame)) + (start_ind - 1)
        return trigger_offset, frame_rate
    except Exception as e:
        print(f"get_trigger_frame_info failed for {box_h5_path}: {e}", flush=True)
        return None, None