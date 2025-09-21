import torch
from torchtnt.framework.callback import Callback
from utils import *
import matplotlib.pyplot as plt
import os
import sys
import csv
import logging
from time import time
from scipy.io import savemat


class ModelCallbacks:
    def __init__(self, model, base_directory, viz_sample, validation):
        self.validation = validation
        self.base_directory = base_directory
        self.viz_sample = viz_sample
        self.model_callbacks = self.config_callbacks(model)
        
        # self.camera_matrices = h5py.File(self.data_path, "r")['/cameras_dlt_array'][:].T
        # self.l2_loss_callback = L2LossCallback(validation, run_path)
        # self.l2_per_point_callback = L2PerPointLossCallback(validation, run_path)
        # self.reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=self.reduce_lr_factor,
        #                                             patience=self.reduce_lr_patience, verbose=1, mode="auto",
        #                                             min_delta=self.reduce_lr_min_delta, cooldown=self.reduce_lr_cooldown,
        #                                             min_lr=self.reduce_lr_min_lr)
        # if self.save_every_epoch:
        #     self.checkpointer = ModelCheckpoint(
        #         filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.keras"),
        #         verbose=1, save_best_only=False)
        # else:
        #     self.checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.keras"), verbose=1,
        #                                         save_best_only=True)
        # self.viz_grid_callback = LambdaCallback(
        #     on_epoch_end=lambda epoch, logs: show_confmap_grid(self.model, *viz_sample, plot=True,
        #                                                        save_path=os.path.join(
        #                                                            run_path,
        #                                                            "viz_confmaps/confmaps_%03d.png" % epoch),
        #                                                        show_figure=False))
        # self.viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(self.model, *viz_sample,
        #                                                                                    save_path=os.path.join(
        #                                                                                        run_path,
        #                                                                                        "viz_pred", "pred_%03d.png" % epoch),
        #                                                                                    show_figure=False))

    def config_callbacks(self, model):
        callbacks = []
        callbacks.append(self.TrainingLogger(log_interval=10))
        callbacks.append(self.EarlyStopping(patience=5))
        callbacks.append(self.L2LossCallback(self.validation, self.base_directory, model))
        callbacks.append(self.L2PerPointLossCallback(self.validation, self.base_directory, model))
        callbacks.append(self.LossHistory(self.base_directory))
        return callbacks
    
    def on_train_start(self):
        for callback in self.model_callbacks:
            if hasattr(callback, 'on_train_start'):
                callback.on_train_start()

    def get_model_callbacks(self):
        return self.model_callbacks
    
    def on_epoch_begin(self, epoch):
        for callback in self.model_callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch=epoch)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.model_callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch=epoch, logs=logs)

    class TrainingLogger:
        def __init__(self, log_interval=10):
            self.log_interval = log_interval
            self.training_logs = []
            logging.basicConfig(
                level=logging.INFO,
                format='%(message)s',
                stream=sys.stdout
            )

        def on_epoch_begin(self, epoch):
            self.epoch_start_time = time()
            logging.info(f"Epoch {epoch + 1} starting.")

        def on_epoch_end(self, epoch, logs=None):
            elapsed_time = time() - self.epoch_start_time
            logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f}s. train loss = {logs['train loss']:.10f}, val loss = {logs['validation loss']:.10f}, lr = {logs['lr']:.10f}.")
            logs['epoch_time'] = elapsed_time  # Add epoch time to logs
            self.training_logs.append(logs)  # Collect training logs

    class L2PerPointLossCallback():

        def __init__(self, validation_data, base_run_directory, model):
            self.box, self.confmaps = validation_data
            self.base_run_directory = base_run_directory
            self.model = model

        def on_epoch_end(self, epoch, n_bins=20,  logs=None):

            self.model.eval()
            device = next(self.model.parameters()).device

            # ---- Predictions ----
            with torch.no_grad():
                x_val = torch.tensor(self.box, dtype=torch.float32).to(device)
                preds = self.model(x_val)

            preds = preds.cpu().numpy()
            pred_peaks = find_peaks(preds)[:, :2, :]  # (B, 2, C)
            gt_peaks   = find_peaks(self.confmaps)[:, :2, :]  # (B, 2, C)

            # Transpose to (B, C, 2)
            pred_peaks = np.transpose(pred_peaks, (0, 2, 1))
            gt_peaks   = np.transpose(gt_peaks, (0, 2, 1))

            # ---- L2 distances per joint ----
            # shape: (B, C) → then transpose to (C, B) for plotting per joint
            l2_per_point_dists = np.linalg.norm(pred_peaks - gt_peaks, axis=-1).T  

            # Handle case when joints are grouped (e.g. 4 cameras)
            num_joints = gt_peaks.shape[1]
            if num_joints > 20:
                cam1, cam2, cam3, cam4 = np.array_split(l2_per_point_dists, 4)
                l2_per_point_dists = np.concatenate((cam1, cam2, cam3, cam4), axis=1)

            num_points = l2_per_point_dists.shape[0]

            # ---- Plot histograms ----
            histogram_path = os.path.join(
                self.base_run_directory,
                "l2_histograms_per_point",
                f"validation_epoch_{epoch + 1}.png"
            )
            os.makedirs(os.path.dirname(histogram_path), exist_ok=True)

            fig, axs = plt.subplots(num_points, 1, figsize=(12, 4 * num_points))

            # If only one point, axs is not an array
            if num_points == 1:
                axs = [axs]

            for i in range(num_points):
                ax = axs[i]
                ax.hist(l2_per_point_dists[i], bins=n_bins, edgecolor="black")
                mean_val = np.mean(l2_per_point_dists[i])
                std_val = np.std(l2_per_point_dists[i])
                ax.set_title(
                    f"Histogram for Point {i + 1} - Mean: {mean_val:.2f}, Std: {std_val:.2f}",
                    fontsize=12
                )
                ax.set_xlabel("L2 distance in pixels", fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)

            plt.tight_layout(pad=3.0)
            plt.savefig(histogram_path)
            plt.close(fig)

    class L2LossCallback():
        def __init__(self, validation_data, base_run_directory, model):
            self.box, self.confmaps = validation_data
            self.base_run_directory = base_run_directory
            self.model = model

        def on_epoch_end(self, epoch, logs=None):
            self.model.eval()
            device = next(self.model.parameters()).device

            # ---- Predictions ----
            with torch.no_grad():
                x_val = torch.tensor(self.box, dtype=torch.float32).to(device)
                preds = self.model(x_val)

            # ---- Find peaks ----
            preds = preds.cpu().numpy()
            pred_peaks = find_peaks(preds)[:, :2, :]  # shape: (B, 2, C)
            pred_peaks = np.transpose(pred_peaks, (0, 2, 1))  # (B, C, 2)

            gt_peaks = find_peaks(self.confmaps)[:, :2, :]
            gt_peaks = np.transpose(gt_peaks, (0, 2, 1))  # (B, C, 2)

            # ---- L2 distances (using numpy) ----
            l2_distances = np.sqrt(np.sum((pred_peaks - gt_peaks) ** 2, axis=2))  # [B, C]
            l2_loss_value = np.mean(l2_distances)

            # ---- Statistics ----
            l2_numpy = l2_distances.flatten()
            std = np.std(l2_numpy)

            if logs is not None:
                logs['val_l2_loss'] = float(l2_loss_value)

            # ---- Plot histogram ----
            plt.figure(figsize=(10, 6))
            plt.hist(l2_numpy, bins=30, alpha=0.75)
            plt.title(
                f"L2 Distance Histogram - Epoch {epoch + 1}\n"
                f"Validation L2 loss: {l2_loss_value:.4f} std: {std:.4f}"
            )
            plt.xlabel("L2 Distance")
            plt.ylabel("Frequency")

            histogram_path = os.path.join(
                self.base_run_directory, "histograms", f"l2_histogram_epoch_{epoch + 1}.png"
            )
            os.makedirs(os.path.dirname(histogram_path), exist_ok=True)
            plt.savefig(histogram_path)
            plt.close()

    class EarlyStopping():
        def __init__(self, patience=3):
            self.patience = patience
            self.counter = 0
            self.best_loss = None

        def on_epoch_end(self, epoch, logs=None):
            current_loss = logs['validation loss']
            if self.best_loss is None or current_loss < self.best_loss:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logging.info("Early stopping triggered.")
                    return True
            return False

    class LossHistory():
        def __init__(self, save_diretory):
            self.save_directory = save_diretory
            self.csv_file_path = os.path.join(self.save_directory, "history.csv")
            self.png_file_path = os.path.join(self.save_directory, "history.png")
            self.mat_file_path = os.path.join(self.save_directory, "history.mat")

        def plot_history(self, history, save_path):
            """ Plots the vision history. """

            loss = [x["train loss"] for x in history]
            val_loss = [x["validation loss"] for x in history]

            plt.figure(figsize=(8, 4))
            plt.plot(loss)
            plt.plot(val_loss)
            plt.semilogy()
            plt.grid()
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(["Training", "Validation"])

            plt.savefig(save_path)
            plt.close()

        def on_train_start(self):
            self.history = []
            if os.path.exists(self.csv_file_path):

                with open(self.csv_file_path, mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        logs = {
                            'train loss': float(row['train loss']),
                            'validation loss': float(row['val loss'])
                        }
                        self.history.append(logs)
                print(f"Resuming history from {self.csv_file_path}, {len(self.history)} epochs loaded.")
            
            else:
                with open(self.csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['epoch', 'train loss', 'val loss'])
                print(f"Starting new history at {self.csv_file_path}.")
        
        def on_epoch_end(self, epoch, logs=None):
            self.history.append(logs.copy())
            savemat(self.mat_file_path,
                    {k: [x[k] for x in self.history] for k in self.history[0].keys()})

            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, logs['train loss'], logs.get('validation loss')])

            self.plot_history(self.history, save_path=self.png_file_path)
