import torch
from utils import torch_find_peaks, VIZ_OUTPUT_DIRECTORY_NAME, show_pred, show_pred_multiple_cameras
import matplotlib.pyplot as plt
import os
import sys
import csv
import numpy as np
import logging
from time import time
from scipy.io import savemat
from constants import MODEL_PER_CAM_PER_WING, MODEL_PER_CAM_PER_WING_UNET, ALL_CAMS_PER_WING

class ModelCallbacks:
    def __init__(self, model, base_directory, viz_sample_list, validation, training=None):
        self.validation = validation
        # (train_box, train_confmap) so the L2 callback can also report the
        # keypoint error on the training set -> a train/val L2 history plot.
        self.training = training
        self.base_directory = base_directory
        self.viz_sample_list = viz_sample_list
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
        callbacks.append(self.L2LossCallback(self.validation, self.base_directory, model, training_data=self.training))
        callbacks.append(self.L2PerPointLossCallback(self.validation, self.base_directory, model))
        callbacks.append(self.LossHistory(self.base_directory))
        callbacks.append(self.VizPredCallback(self.viz_sample_list, self.base_directory, model))
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

    @staticmethod
    def compute_l2_pixel_error(model, box, confmaps, batch_size=16):
        """Mean L2 keypoint error (in pixels) between the predicted and the
        ground-truth heatmap peaks.

        Inference is run in mini-batches so the whole set never has to sit on
        the GPU at once (the training set is ~9x the validation set).

        Returns:
            (mean_l2, flat_l2_distances) where flat_l2_distances is a 1-D array
            of per-(sample, joint) distances, handy for histogramming.
        """
        model.eval()
        device = next(model.parameters()).device
        dists = []
        with torch.no_grad():
            for start in range(0, len(box), batch_size):
                x = torch.tensor(box[start:start + batch_size], dtype=torch.float32).to(device)
                preds = model(x).cpu().numpy()
                # (b, 2, C) -> (b, C, 2)
                pred_peaks = np.transpose(torch_find_peaks(preds)[:, :2, :], (0, 2, 1))
                gt_peaks = np.transpose(torch_find_peaks(confmaps[start:start + batch_size])[:, :2, :], (0, 2, 1))
                dists.append(np.linalg.norm(pred_peaks - gt_peaks, axis=2))  # (b, C)
        l2 = np.concatenate(dists, axis=0)  # (B, C)
        return float(l2.mean()), l2.flatten()

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
            pred_peaks = torch_find_peaks(preds)[:, :2, :]  # (B, 2, C)
            gt_peaks   = torch_find_peaks(self.confmaps)[:, :2, :]  # (B, 2, C)

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
        def __init__(self, validation_data, base_run_directory, model, training_data=None):
            self.box, self.confmaps = validation_data
            # training_data is optional: when provided, the training-set L2 is
            # logged alongside the validation L2 so LossHistory can plot both.
            self.train_box, self.train_confmaps = training_data if training_data is not None else (None, None)
            self.base_run_directory = base_run_directory
            self.model = model

        def on_epoch_end(self, epoch, logs=None):
            # ---- Validation L2 (mean pixel error + per-sample distances) ----
            val_l2, val_l2_flat = ModelCallbacks.compute_l2_pixel_error(
                self.model, self.box, self.confmaps
            )
            if logs is not None:
                logs['val_l2_loss'] = val_l2

            # ---- Training L2 (same metric on the un-augmented train set) ----
            if self.train_box is not None:
                train_l2, _ = ModelCallbacks.compute_l2_pixel_error(
                    self.model, self.train_box, self.train_confmaps
                )
                if logs is not None:
                    logs['train_l2_loss'] = train_l2

            # ---- Plot validation histogram ----
            std = np.std(val_l2_flat)
            plt.figure(figsize=(10, 6))
            plt.hist(val_l2_flat, bins=30, alpha=0.75)
            plt.title(
                f"L2 Distance Histogram - Epoch {epoch + 1}\n"
                f"Validation L2 loss: {val_l2:.4f} std: {std:.4f}"
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
            self.l2_png_file_path = os.path.join(self.save_directory, "history_l2.png")
            self.mat_file_path = os.path.join(self.save_directory, "history.mat")

        def plot_history(self, history, save_path):
            """ Plots the vision history. """

            train_loss = [x["train loss"] for x in history]
            val_loss = [x["validation loss"] for x in history]

            plt.figure(figsize=(8, 4))
            plt.plot(train_loss)
            plt.plot(val_loss)
            plt.semilogy()
            plt.grid()
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(["Training", "Validation"])

            plt.savefig(save_path)
            plt.close()

        def plot_l2_history(self, history, save_path):
            """Plots the keypoint L2 pixel-error history (train + val).

            The L2 error is the quantity we actually care about, so unlike the
            background-dominated heatmap loss it is plotted on a linear y-axis.
            Epochs missing a value (e.g. rows restored from an older resume CSV
            that predates L2 logging) are skipped rather than breaking the plot.
            """
            def series(key):
                xs = [i for i, x in enumerate(history) if x.get(key) is not None]
                ys = [history[i][key] for i in xs]
                return xs, ys

            train_x, train_y = series("train_l2_loss")
            val_x, val_y = series("val_l2_loss")
            if not train_y and not val_y:
                return  # nothing logged yet, don't emit an empty figure

            plt.figure(figsize=(8, 4))
            legend = []
            if train_y:
                plt.plot(train_x, train_y)
                legend.append("Training")
            if val_y:
                plt.plot(val_x, val_y)
                legend.append("Validation")
            plt.grid()
            plt.xlabel("Epochs")
            plt.ylabel("L2 error (pixels)")
            plt.legend(legend)

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
                        # L2 columns are optional: only present in CSVs written
                        # by this (or a newer) version, so restore them if there.
                        if row.get('train l2') not in (None, ''):
                            logs['train_l2_loss'] = float(row['train l2'])
                        if row.get('val l2') not in (None, ''):
                            logs['val_l2_loss'] = float(row['val l2'])
                        self.history.append(logs)
                print(f"Resuming history from {self.csv_file_path}, {len(self.history)} epochs loaded.")

            else:
                with open(self.csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['epoch', 'train loss', 'val loss', 'train l2', 'val l2'])
                print(f"Starting new history at {self.csv_file_path}.")

        def on_epoch_end(self, epoch, logs=None):
            self.history.append(logs.copy())
            savemat(self.mat_file_path,
                    {k: [x[k] for x in self.history] for k in self.history[0].keys()})

            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, logs['train loss'], logs.get('validation loss'),
                                 logs.get('train_l2_loss'), logs.get('val_l2_loss')])

            self.plot_history(self.history, save_path=self.png_file_path)
            self.plot_l2_history(self.history, save_path=self.l2_png_file_path)

    class VizPredCallback():
        def __init__(self, sample_confmaps_list, save_directory, model):
            self.samples, self.confmaps = sample_confmaps_list
            self.save_directory = os.path.join(save_directory, VIZ_OUTPUT_DIRECTORY_NAME)
            self.model = model
            self.create_sample_directories()

        def create_sample_directories(self):
            for i in range(len(self.samples)):
                sample_dir = os.path.join(self.save_directory, f"sample_{i}")
                os.makedirs(sample_dir, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            for i, (sample, confmap) in enumerate(zip(self.samples, self.confmaps)):
                sample_save_dir = os.path.join(self.save_directory, f"sample_{i}")
                model_type = self.model.get_model_type()
                if model_type == MODEL_PER_CAM_PER_WING or model_type == MODEL_PER_CAM_PER_WING_UNET:
                    show_pred(
                        self.model,
                        sample,
                        confmap,
                        epoch_num=epoch,
                        save_directory=sample_save_dir,
                    )
                elif model_type == ALL_CAMS_PER_WING:
                    show_pred_multiple_cameras(
                        self.model,
                        sample,
                        confmap,
                        epoch_num=epoch,
                        save_directory=sample_save_dir,
                        num_cameras=4,
                        num_points=10
                    )
