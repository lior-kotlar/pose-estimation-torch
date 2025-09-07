import torch
from torchtnt.framework.callback import Callback
from utils import *
import matplotlib.pyplot as plt
import os
import h5py


class L2LossCallback(Callback):
    def __init__(self, validation_data, base_run_directory, model):
        super().__init__()
        self.box, self.confmaps = validation_data
        self.base_run_directory = base_run_directory
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        self.model.eval()
        device = next(self.model.parameters()).device

        # ---- Predictions ----
        with torch.no_grad():
            x_val = torch.tensor(self.box, dtype=torch.float32).to(device)
            true_confmaps = torch.tensor(self.confmaps, dtype=torch.float32).to(device)

            preds = self.model(x_val)

        # ---- Find peaks ----
        pred_peaks = find_peaks(preds)
        pred_peaks = pred_peaks[:, :2, :].permute(0, 2, 1)  # [B, C, 2]
        gt_peaks   = find_peaks(true_confmaps)
        gt_peaks   = gt_peaks[:, :2, :].permute(0, 2, 1)    # [B, C, 2]

        # ---- L2 distances ----
        l2_distances = torch.sqrt(torch.sum((pred_peaks - gt_peaks) ** 2, dim=2))  # [B, C]
        l2_loss = torch.mean(l2_distances)

        # ---- Statistics ----
        l2_numpy = l2_distances.cpu().numpy().flatten()
        std = np.std(l2_numpy)

        if logs is not None:
            logs['val_l2_loss'] = l2_loss.item()

        # ---- Plot histogram ----
        plt.figure(figsize=(10, 6))
        plt.hist(l2_numpy, bins=30, alpha=0.75)
        plt.title(f"L2 Distance Histogram - Epoch {epoch + 1}\n"
                f"Validation L2 loss: {l2_loss.item():.4f} std: {std:.4f}")
        plt.xlabel("L2 Distance")
        plt.ylabel("Frequency")

        histogram_path = os.path.join(self.base_run_directory, "histograms", f"l2_histogram_epoch_{epoch + 1}.png")
        os.makedirs(os.path.dirname(histogram_path), exist_ok=True)
        plt.savefig(histogram_path)
        plt.close()

class L2PerPointLossCallback(Callback):

    def __init__(self, validation_data, base_run_directory, model):
        super().__init__()
        
        self.base_run_directory = base_run_directory
        self.model = model

    def on_epoch_end(self, epoch, logs=None, n_bins=20):

        self.model.eval()
        device = next(self.model.parameters()).device

        # ---- Predictions ----
        with torch.no_grad():
            x_val = torch.tensor(self.box, dtype=torch.float32).to(device)
            true_confmaps = torch.tensor(self.confmaps, dtype=torch.float32).to(device)
            preds = self.model(x_val)

        pred_peaks = find_peaks(preds)
        pred_peaks = pred_peaks[:, :2, :].permute(0, 2, 1)

        gt_peaks = find_peaks(true_confmaps)
        gt_peaks = gt_peaks[:, :2, :].permute(0, 2, 1)

        pred_peaks = pred_peaks.numpy()
        gt_peaks = gt_peaks.numpy()
        pred_peaks = np.transpose(pred_peaks, (0, 2, 1))
        gt_peaks = np.transpose(gt_peaks, (0, 2, 1))
        num_joints = gt_peaks.shape[1]
        l2_per_point_dists = np.linalg.norm(pred_peaks - gt_peaks, axis=-1).T
        if num_joints > 20:
            cam1, cam2, cam3, cam4 = np.array_split(l2_per_point_dists, 4)
            l2_per_point_dists = np.concatenate((cam1, cam2, cam3, cam4), axis=1)
        num_points = l2_per_point_dists.shape[0]
        histogram_path = os.path.join(self.base_run_directory, 'l2_histograms_per_point', f'validation_epoch_{epoch + 1}.png')
        fig, axs = plt.subplots(num_points, 1, figsize=(12, 4 * num_points))
        for i in range(num_points):
            ax = axs[i]
            ax.hist(l2_per_point_dists[i], bins=n_bins, edgecolor='black')
            mean_val = np.mean(l2_per_point_dists[i])
            std_val = np.std(l2_per_point_dists[i])
            ax.set_title(f'Histogram for Point {i + 1} - Mean: {mean_val:.2f}, Std: {std_val:.2f}', fontsize=12)
            ax.set_xlabel('L2 distance in pixels', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
        plt.tight_layout(pad=3.0)
        plt.savefig(histogram_path)
        plt.close(fig)

class Callbacks:
    def __init__(self, config, run_path, model, viz_sample, validation):
        self.model = model
        self.reduce_lr_factor = config["reduce_lr_factor"]
        self.reduce_lr_patience = config["reduce_lr_patience"]
        self.reduce_lr_min_delta = config["reduce_lr_min_delta"]
        self.reduce_lr_cooldown = config["reduce_lr_cooldown"]
        self.reduce_lr_min_lr = config["reduce_lr_min_lr"]
        self.save_every_epoch = bool(config["save_every_epoch"])
        self.data_path = config["data_path"]
        self.camera_matrices = h5py.File(self.data_path, "r")['/cameras_dlt_array'][:].T
        self.l2_loss_callback = L2LossCallback(validation, run_path)
        self.l2_per_point_callback = L2PerPointLossCallback(validation, run_path)
        self.reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=self.reduce_lr_factor,
                                                    patience=self.reduce_lr_patience, verbose=1, mode="auto",
                                                    min_delta=self.reduce_lr_min_delta, cooldown=self.reduce_lr_cooldown,
                                                    min_lr=self.reduce_lr_min_lr)
        if self.save_every_epoch:
            self.checkpointer = ModelCheckpoint(
                filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.keras"),
                verbose=1, save_best_only=False)
        else:
            self.checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.keras"), verbose=1,
                                                save_best_only=True)
        self.viz_grid_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: show_confmap_grid(self.model, *viz_sample, plot=True,
                                                               save_path=os.path.join(
                                                                   run_path,
                                                                   "viz_confmaps/confmaps_%03d.png" % epoch),
                                                               show_figure=False))
        self.viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(self.model, *viz_sample,
                                                                                           save_path=os.path.join(
                                                                                               run_path,
                                                                                               "viz_pred", "pred_%03d.png" % epoch),
                                                                                           show_figure=False))


    def get_callbacks(self):
        return [self.reduce_lr_callback,
                self.checkpointer,
                self.viz_pred_callback,
                self.l2_loss_callback,
                self.l2_per_point_callback]