import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d, binary_dilation
from skimage.morphology import disk, erosion, dilation
from validation import Validation
import json
import h5py
import numpy as np
import os
from multiprocessing import Pool, cpu_count
# from tensorflow import keras
from tensorflow.keras.layers import Lambda
import tensorflow.keras.config as tfconfig
from scipy.interpolate import make_smoothing_spline
from skimage import util, measure
import tensorflow as tf
from scipy.spatial.distance import cdist
import torch
from itertools import combinations
import math
from scipy.io import loadmat
# imports of the wings1 detection
from time import time
from ultralytics import YOLO
# import open3d as o3d
import scipy
from scipy.signal import medfilt
from scipy.ndimage import binary_dilation, binary_closing, center_of_mass, shift, gaussian_filter, binary_opening
from datetime import date
import shutil
from skimage.morphology import convex_hull_image
from torchvision import transforms
import keras
from keras.layers import LeakyReLU

# from scipy.spatial.distance import pdist
# from scipy.ndimage.measurements import center_of_mass
# from scipy.spatial import ConvexHull
# import matplotlib
# import cv2
# import preprocessor
from constants import *
import predictions_2Dto3D
import sys
from predictions_2Dto3D import From2Dto3D

sys.path.append(r'C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\predict_2D_pytorch')
from BoxSparse import BoxSparse
from traingulate import Triangulate

initial_gpus = tf.config.list_physical_devices('GPU')
# Hide all GPUs from TensorFlow
# tf.config.set_visible_devices([], 'GPU')
# print("GPUs have been hidden.")
print(f"Initially available GPUs: {initial_gpus}", flush=True)

DETECT_WINGS_CPU = False


WHICH_TO_FLIP = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype(bool)
ALL_COUPLES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
LEFT = 0
RIGHT = 1


class Predictor2D:
    def __init__(self, configuration_path, load_box_from_sparse=False, is_masked=False):
        self.points_3D_smoothed = None
        self.points_3D = None
        self.saved_box_dir = None
        self.body_sizes = None
        self.neto_wings_sparse = None
        self.wings_size = None
        self.body_masks_sparse = None
        self.run_path = None
        self.triangulation_errors = None
        self.reprojection_errors = None
        self.points_3D_all = None
        self.conf_preds = None
        self.preds_2D = None
        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.software = config['software']
            self.triangulate = Triangulate(self.config)
            self.box_path = config["box path"]
            self.wings_pose_estimation_model_path = config["wings pose estimation model path"]
            self.wings_pose_estimation_model_path_second_pass = config["wings pose estimation model path second path"]
            self.head_tail_pose_estimation_model_path = config["head tail pose estimation model path"]
            # self.out_path = config["out path"]
            self.wings_detection_model_path = config["wings detection model path"]
            self.model_type = config["model type"]
            self.model_type_second_pass = config["model type second pass"]
            self.is_video = bool(config["is video"])
            self.batch_size = config["batch size"]
            self.points_to_predict = config["body parts to predict"]
            self.num_cams = config["number of cameras"]
            self.num_times_channels = config["number of time channels"]
            self.mask_increase_initial = config["mask increase initial"]
            self.mask_increase_reprojected = config["mask increase reprojected"]
            self.use_reprojected_masks = bool(config["use reprojected masks"])
            self.predict_again_using_3D_consistent = config["predict again 3D consistent"]
            self.base_output_path = config["base output path"]
            self.json_2D_3D_path = config["2D to 3D config path"]
        self.load_from_sparse = load_box_from_sparse
        if not load_box_from_sparse:
            print("creating sparse box object")
            self.box_sparse = BoxSparse(self.box_path, is_masked=is_masked)
            box = self.box_sparse.retrieve_dense_box()
            print("finish creating sparse box object")
        else:
            self.load_preprocessed_box()
        self.masks_flag = is_masked
        # Visualizer.display_movie_from_box(np.copy(self.box))
        self.cropzone = self.get_cropzone()
        self.im_size = self.box_sparse.shape[2]
        self.num_frames = self.box_sparse.shape[0]
        self.num_pass = 0
        if self.software == 'tensorflow':
            return_model_peaks = True
            if self.model_type == ALL_CAMS_PER_WING or self.model_type == ALL_CAMS_ALL_POINTS:
                return_model_peaks = False
            self.wings_pose_estimation_model = \
                Predictor2D.get_pose_estimation_model_tensorflow(self.wings_pose_estimation_model_path, return_model_peaks)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                torch.zeros(4).cuda()
            except:
                print("No GPU found, doesnt use cuda")
            if torch.cuda.is_available():
                print("**************** CUDA is available. Using GPU. ****************", flush=True)
            else:
                print("**************** CUDA is not available. Using CPU. ****************", flush=True)
            self.wings_pose_estimation_model = \
                self.get_pose_estimation_model_pytorch(self.wings_pose_estimation_model_path)
            self.wings_pose_estimation_model = self.wings_pose_estimation_model.to(self.device)
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.wings_detection_model = self.get_wings_detection_model()
        self.scores = np.zeros((self.num_frames, self.num_cams, 2))
        self.predict_method = self.choose_predict_method()

        self.run_name = self.get_run_name()

        self.total_runtime = None
        self.prediction_runtime = None
        self.predicted_points = None
        self.out_path_h5 = None

        self.num_joints = 18
        self.left_mask_ind, self.right_mask_ind = 3, 4
        self.image_size = self.box_sparse.shape[-2]
        self.num_time_channels = 3
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.wings_pnts_inds = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.left_mask_ind = 3
        self.right_mask_ind = 4

    def run_predict_2D(self, save=True):
        """
        creates an array of pose estimation predictions
        """
        t0 = time()
        self.run_path = self.create_run_folders()
        if not self.load_from_sparse:
            if not self.masks_flag:
                print("cleaning the images")
                self.clean_images()
                print("aligning time channels")
                self.align_time_channels()

            print("find body masks")
            self.set_body_masks()
            print("preprocessing masks")
            self.preprocess_masks()
            print("finding wings sizes")
            self.get_neto_wings_masks()

        preprocessing_time = time() - t0
        preds_time = time()
        print("preprocess [%.1fs]" % preprocessing_time)
        self.predicted_points = self.predict_method()
        self.preds_2D = self.predicted_points[..., :-1]
        self.conf_preds = self.predicted_points[..., -1]

        print("finish predict", flush=True)
        self.prediction_runtime = 0
        if self.is_video:
            print("enforcing 3D consistency", flush=True)
            self.enforce_3D_consistency()
        print("done")

        # box = self.box_sparse.retrieve_dense_box()
        # points_2D = self.preds_2D
        # from visualize import Visualizer
        # Visualizer.show_predictions_all_cams(box, points_2D)

        print("predicting 3D points", flush=True)
        self.points_3D_all, self.reprojection_errors, self.triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone)

        print("saving", flush=True)
        if save:
            self.save_predictions_to_h5()
        # box = self.box_sparse.retrieve_dense_box()
        # Visualizer.show_predictions_all_cams(box, self.predicted_points)
        print("done saving", flush=True)
        if self.predict_again_using_3D_consistent:
            print("starting the reprojected masks creation")
            self.num_pass += 1
            self.model_type = self.model_type_second_pass
            self.predict_method = self.choose_predict_method()
            return_model_peaks = False if self.model_type == ALL_CAMS_PER_WING or self.model_type == ALL_CAMS_ALL_POINTS else True
            self.wings_pose_estimation_model = \
                self.get_pose_estimation_model_tensorflow(self.wings_pose_estimation_model_path_second_pass,
                                               return_model_peaks=return_model_peaks)
            if self.use_reprojected_masks:
                # points_3D = self.get_points_3D(alpha=None)
                # smoothed_3D = self.smooth_3D_points(points_3D)
                self.get_reprojection_masks(self.points_3D_smoothed, self.mask_increase_reprojected,
                                            deside_between_new_and_orig=False)
                print("created reprojection masks", flush=True)
                print("predicting second round, now with reprojected masks", flush=True)
            self.predicted_points = self.predict_method()
            self.preds_2D = self.predicted_points[..., :-1]
            self.conf_preds = self.predicted_points[..., -1]
            self.points_3D_all, self.reprojection_errors = self.get_all_3D_pnts_all_cameras_combinations(self.preds_2D,
                                                                                                         self.cropzone)
            print("saving")
            self.save_predictions_to_h5()

        # Visualizer.show_predictions_all_cams(self.box, self.predicted_points)
        self.prediction_runtime = time() - preds_time
        self.total_runtime = time() - t0
        print("Predicted [%.1fs]" % self.prediction_runtime)
        print("Prediction performance: %.3f FPS" % (self.num_frames * self.num_cams / self.prediction_runtime))

    def create_base_box(self):
        print("cleaning the images", flush=True)
        self.clean_images()
        print("aligning time channels", flush=True)
        self.align_time_channels()
        print("find body masks", flush=True)
        self.set_body_masks()
        print("preprocessing masks", flush=True)
        self.preprocess_masks()
        print("finding wings sizes", flush=True)
        self.get_neto_wings_masks()

    def save_base_box(self):
        print("saving box")
        # save the sparse box, neto wings sparse, body masks sparse, body_sizes, wings sizes
        mov_dir_name = os.path.dirname(self.box_path)
        self.saved_box_dir = os.path.join(mov_dir_name, "saved_box_dir")
        if os.path.exists(self.saved_box_dir):
            shutil.rmtree(self.saved_box_dir)
        os.makedirs(self.saved_box_dir)

        # save sparse box
        box_save_name = os.path.join(self.saved_box_dir, "box.h5")
        self.box_sparse.save_to_scipy_sparse_format(box_save_name)

        # save neto wings
        wings_save_name = os.path.join(self.saved_box_dir, "neto_wings.h5")
        self.neto_wings_sparse.save_to_scipy_sparse_format(wings_save_name)

        # save body masks
        body_save_name = os.path.join(self.saved_box_dir, "body_masks.h5")
        self.body_masks_sparse.save_to_scipy_sparse_format(body_save_name)

        # save arrays of body sizes
        body_sizes_path = os.path.join(self.saved_box_dir, "body_sizes.npy")
        np.save(body_sizes_path, self.body_sizes)

        # save arrays of wings_size
        wings_size_path = os.path.join(self.saved_box_dir, "wings_size.npy")
        np.save(wings_size_path, self.wings_size)

    def load_preprocessed_box(self):
        mov_dir_name = os.path.dirname(self.box_path)
        self.saved_box_dir = os.path.join(mov_dir_name, "saved_box_dir")

        # load box
        box_save_name = os.path.join(self.saved_box_dir, "box.h5")
        self.box_sparse = BoxSparse(load_from_sparse=True, sparse_path=box_save_name)

        # load neto wings
        wings_save_name = os.path.join(self.saved_box_dir, "neto_wings.h5")
        self.neto_wings_sparse = BoxSparse(load_from_sparse=True, sparse_path=wings_save_name)

        # load body masks
        body_save_name = os.path.join(self.saved_box_dir, "body_masks.h5")
        self.body_masks_sparse = BoxSparse(load_from_sparse=True, sparse_path=body_save_name)

        # load body sizes
        body_sizes_path = os.path.join(self.saved_box_dir, "body_sizes.npy")
        self.body_sizes = np.load(body_sizes_path)

        # load wings sizes
        wings_size_path = os.path.join(self.saved_box_dir, "wings_size.npy")
        self.wings_size = np.load(wings_size_path)

    @staticmethod
    def get_median_point(all_points_3D):
        median = np.median(all_points_3D, axis=2)
        return median

    @staticmethod
    def get_validation_score(points_3D):
        return Validation.get_wings_distances_variance(points_3D)[0]

    @staticmethod
    def choose_best_reprojection_error_points(points_3D_all, reprojection_errors_all):
        num_frames, num_joints, _, _ = points_3D_all.shape
        points_3D = np.zeros((num_frames, num_joints, 3))
        reprojection_errors_chosen = np.zeros((num_frames, num_joints))
        for frame in range(num_frames):
            for joint in range(num_joints):
                candidates = points_3D_all[frame, joint, ...]
                best_candidate_ind = np.argmin(reprojection_errors_all[frame, joint, ...])
                reprojection_errors_chosen[frame, joint] = reprojection_errors_all[frame, joint, best_candidate_ind]
                point_3d = candidates[best_candidate_ind]
                points_3D[frame, joint, :] = point_3d
        return points_3D, reprojection_errors_chosen

    def get_points_3D(self, alpha=None):
        if alpha is None:
            scores1 = []
            scores2 = []
            alphas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
            print("start choosing 3D points")
            for alpha in alphas:
                points_3D_i = self.choose_best_score_2_cams(alpha=alpha)
                smoothed_3D_i = self.smooth_3D_points(points_3D_i)
                score1 = self.get_validation_score(points_3D_i)
                score2 = self.get_validation_score(smoothed_3D_i)
                # print(f"alpha = {alpha} score1 is {score1}, score2 is {score2}")
                scores1.append(score1)
                scores2.append(score2)
            min_alpha_ind = np.argmin(scores2)
            min_alpha = alphas[min_alpha_ind]
            points_3D_chosen = self.choose_best_score_2_cams(min_alpha)
            print("finish choosing 3D points for 3D consistency")
            return points_3D_chosen
        else:
            points_3D_out = self.choose_best_score_2_cams(alpha=alpha)
            # smoothed_3D_i = self.smooth_3D_points(points_3D_i)
            # score1 = self.get_validation_score(points_3D_i)
            # score2 = self.get_validation_score(smoothed_3D_i)
            # print(f"alpha = {alpha} score1 is {score1}, score2 is {score2}")
            return points_3D_out

    @staticmethod
    def smooth_3D_points(points_3D, lam=1):
        points_3D_smoothed = np.zeros_like(points_3D)
        num_joints = points_3D_smoothed.shape[1]
        num_points = len(points_3D)
        lam = None
        for pnt in range(num_joints):
            for axis in range(3):
                # print(pnt, axis)
                vals = points_3D[:, pnt, axis]
                # set lambda as the regularising parameters: smoothing vs close to data
                # lam = 300 if pnt in SIDE_POINTS else None
                A = np.arange(vals.shape[0])
                if pnt in BODY_POINTS:
                    vals = medfilt(vals, kernel_size=11)
                    W = np.ones_like(vals)
                else:
                    filtered_data = medfilt(vals, kernel_size=3)
                    # Compute the absolute difference between the original data and the filtered data
                    diff = np.abs(vals - filtered_data)
                    # make the diff into weights in [0,1]
                    diff = diff / np.max(diff)
                    W = 1 - diff
                    W[W == 0] = 0.00001
                spline = make_smoothing_spline(A, vals, w=W, lam=lam)
                smoothed = spline(A)
                points_3D_smoothed[:, pnt, axis] = smoothed
        return points_3D_smoothed

    def choose_best_score_2_cams(self, alpha=0.7):
        envelope_2D = self.get_derivative_envelope_2D()
        points_3D = np.zeros((self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            # start with head and tail points
            for ind in self.head_tail_inds:
                body_sizes = self.body_sizes[frame, :]
                candidates = self.points_3D_all[frame, ind, :, :]
                max_size = np.max(body_sizes)
                body_sizes_score = body_sizes / max_size
                noise = envelope_2D[frame, :, ind] / np.max(envelope_2D[frame, :, ind])
                noise_score = 1 - noise
                scores = body_sizes_score + noise_score
                # scores = noise_score  # todo now the score is only noise
                # scores = scores * visibility_score
                cameras_ind = np.sort(np.argpartition(scores, -2)[-2:])
                best_pair_ind = self.triangulate.all_couples.index(tuple(cameras_ind))
                best_3D_point = candidates[best_pair_ind]
                points_3D[frame, ind, :] = best_3D_point

            for wing_num in range(2):
                for pnt_num in self.wings_pnts_inds[wing_num, :]:
                    candidates = self.points_3D_all[frame, pnt_num, :, :]
                    wings_size = self.wings_size[frame, :, wing_num]
                    max_size = np.max(wings_size)
                    masks_sizes_score = wings_size / max_size
                    noise = envelope_2D[frame, :, pnt_num] / np.max(envelope_2D[frame, :, pnt_num])
                    noise_score = 1 - noise
                    scores = alpha * masks_sizes_score + (1 - alpha) * noise_score
                    # scores = scores * visibility_score
                    cameras_ind = np.sort(np.argpartition(scores, -2)[-2:])
                    best_pair_ind = self.triangulate.all_couples.index(tuple(cameras_ind))
                    best_3D_point = candidates[best_pair_ind]
                    points_3D[frame, pnt_num, :] = best_3D_point
        # now find the
        # points_3D[:, self.head_tail_inds, :] = self.choose_best_reprojection_error_points()[:, self.head_tail_inds, :]
        return points_3D

    def get_derivative_envelope_2D(self):
        derivative_2D = self.get_2D_derivatives()
        envelope_2D = np.zeros(shape=self.preds_2D.shape[:-1])
        for cam in range(self.num_cams):
            for joint in range(self.num_joints):
                signal = derivative_2D[:, cam, joint]
                # Define the cutoff frequency for the low-pass filter (between 0 and 1)
                # Calculate the analytic signal, from which the envelope is the magnitude
                analytic_signal = hilbert(signal)
                envelope = np.abs(analytic_signal)
                smooth_envelope = gaussian_filter1d(envelope, 0.7)
                envelope_2D[:, cam, joint] = smooth_envelope
        return envelope_2D

    def get_2D_derivatives(self):
        derivative_2D = np.zeros(shape=self.preds_2D.shape[:-1])
        derivative_2D[1:, ...] = np.linalg.norm(self.preds_2D[1:, ...] - self.preds_2D[:-1, ...], axis=-1)
        return derivative_2D

    def clean_images(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for channel in range(self.num_times_channels):
                    image = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel)
                    # image = self.box[frame, cam, :, :, channel]
                    binary = np.where(image >= 0.1, 1, 0)
                    label = measure.label(binary)
                    props = measure.regionprops(label)
                    sizes = [prop.area for prop in props]
                    largest = np.argmax(sizes)
                    fly_component = np.where(label == largest + 1, 1, 0)
                    image = image * fly_component
                    # self.box[frame, cam, :, :, channel] = image
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, channel, image)

    def enforce_3D_consistency(self):
        wings_size_togather = self.wings_size[..., LEFT] * self.wings_size[..., RIGHT]
        for frame in range(self.num_frames):
            wings_score = wings_size_togather[frame]
            chosen_camera = np.argmax(wings_score)
            cameras_to_check = np.arange(0, 4)
            cameras_to_check = cameras_to_check[np.where(cameras_to_check != chosen_camera)]
            # step 1
            if frame > 0:
                switch_flag = self.deside_if_switch(chosen_camera, frame)
                if switch_flag:
                    self.flip_camera(chosen_camera, frame)

            # step 2
            cameras_to_flip = self.find_which_cameras_to_flip(cameras_to_check, frame)
            # print(f"frame {frame}, camera to flip {cameras_to_flip}")
            for cam in cameras_to_flip:
                self.flip_camera(cam, frame)

        # might cause problems
        self.enforce_3D_left_right_consistency()

    def enforce_3D_left_right_consistency(self):
        self.points_3D_all, self.reprojection_errors, self.triangulation_errors = (
            self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone))
        _, points_3D, _, _ = Predictor2D.find_3D_points_optimize_neighbors([self.points_3D_all])
        right_points = np.zeros((self.num_frames, len(self.right_inds), 3))
        left_points = np.zeros((self.num_frames, len(self.left_inds), 3))

        # initialize
        right_points[0, ...] = points_3D[0, self.right_inds]
        left_points[0, ...] = points_3D[0, self.left_inds]

        for frame in range(1, self.num_frames):
            cur_left_points = points_3D[frame, self.left_inds, :]
            cur_right_points = points_3D[frame, self.right_inds, :]

            prev_left_points = left_points[frame - 1, :]
            prev_right_points = right_points[frame - 1, :]

            l2l_dist = np.linalg.norm(cur_left_points - prev_left_points)
            r2r_dist = np.linalg.norm(cur_right_points - prev_right_points)
            r2l_dist = np.linalg.norm(cur_right_points - prev_left_points)
            l2r_dist = np.linalg.norm(cur_left_points - prev_right_points)
            do_switch = l2l_dist + r2r_dist > r2l_dist + l2r_dist

            if do_switch:
                right_points[frame] = cur_left_points
                left_points[frame] = cur_right_points
                for cam in range(self.num_cams):
                    self.flip_camera(cam, frame)
            else:
                right_points[frame] = cur_right_points
                left_points[frame] = cur_left_points

    def get_reprojection_masks(self, points_3D, extend_mask_radius=7, deside_between_new_and_orig=False):
        points_2D_reprojected = self.triangulate.get_reprojections(points_3D, self.cropzone)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=1)
                for wing in range(2):
                    points_inds = self.wings_pnts_inds[wing, :]
                    mask = np.zeros((self.image_size, self.image_size))
                    wing_pnts = np.round(points_2D_reprojected[frame, cam, points_inds, :]).astype(int)
                    wing_pnts[wing_pnts >= self.image_size] = self.image_size - 1
                    wing_pnts[wing_pnts < 0] = 0
                    mask[wing_pnts[:, 1], wing_pnts[:, 0]] = 1
                    mask = convex_hull_image(mask)  # todo switch
                    mask = binary_dilation(mask, iterations=extend_mask_radius)
                    mask = np.logical_and(mask, fly)
                    mask = binary_dilation(mask, iterations=2)
                    orig_mask = self.box_sparse.get_frame_camera_channel_dense(frame,
                                                                               cam,
                                                                               channel_idx=self.num_times_channels + wing)
                    if deside_between_new_and_orig:
                        if np.count_nonzero(np.logical_and(mask, fly)) > np.count_nonzero(np.logical_and(orig_mask, fly)):
                            self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + wing, mask)
                        else:
                            self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + wing,
                                                                           orig_mask)
                    else:
                        self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + wing, mask)

    def deside_if_switch(self, chosen_camera, frame):
        cur_left_points = self.preds_2D[frame, chosen_camera, self.left_inds, :]
        cur_right_points = self.preds_2D[frame, chosen_camera, self.right_inds, :]
        prev_left_points = self.preds_2D[frame - 1, chosen_camera, self.left_inds, :]
        prev_right_points = self.preds_2D[frame - 1, chosen_camera, self.right_inds, :]
        l2l_dist = np.linalg.norm(cur_left_points - prev_left_points)
        r2r_dist = np.linalg.norm(cur_right_points - prev_right_points)
        r2l_dist = np.linalg.norm(cur_right_points - prev_left_points)
        l2r_dist = np.linalg.norm(cur_left_points - prev_right_points)
        do_switch = l2l_dist + r2r_dist > r2l_dist + l2r_dist
        return do_switch

    def get_all_3D_pnts_pairs(self, points_2D, cropzone):
        points_3D_all, reprojection_errors, triangulation_errors = \
            self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)
        return points_3D_all, reprojection_errors, triangulation_errors

    def get_all_3D_pnts_all_cameras_combinations(self, points_2D, cropzone):
        points_3D_all, reprojection_errors = self.triangulate.triangulate_points_all_possible_views(points_2D, cropzone)
        return points_3D_all, reprojection_errors

    def find_which_cameras_to_flip(self, cameras_to_check, frame):
        num_of_options = len(WHICH_TO_FLIP)
        switch_scores = np.zeros(num_of_options, )
        cropzone = self.cropzone[frame][np.newaxis, ...]
        for i, option in enumerate(WHICH_TO_FLIP):
            points_2D = np.copy(self.preds_2D[frame])
            cameras_to_flip = cameras_to_check[option]
            for cam in cameras_to_flip:
                left_points = points_2D[cam, self.left_inds, :]
                right_points = points_2D[cam, self.right_inds, :]
                points_2D[cam, self.left_inds, :] = right_points
                points_2D[cam, self.right_inds, :] = left_points
            points_2D = points_2D[np.newaxis, ...]
            points_3D_all, reprojection_errors, _ = self.get_all_3D_pnts_pairs(points_2D, cropzone)
            # _, reprojection_errors_chosen = self.choose_best_reprojection_error_points(points_3D_all, reprojection_errors)
            score = np.mean(reprojection_errors)
            switch_scores[i] = score
        cameras_to_flip = cameras_to_check[WHICH_TO_FLIP[np.argmin(switch_scores)]]
        return cameras_to_flip

    def flip_camera(self, camera_to_flip, frame):
        left_points = self.preds_2D[frame, camera_to_flip, self.left_inds, :]
        right_points = self.preds_2D[frame, camera_to_flip, self.right_inds, :]
        self.preds_2D[frame, camera_to_flip, self.left_inds, :] = right_points
        self.preds_2D[frame, camera_to_flip, self.right_inds, :] = left_points
        # switch train_masks in box
        left_mask = self.box_sparse.get_frame_camera_channel_dense(frame, camera_to_flip, self.left_mask_ind)
        right_mask = self.box_sparse.get_frame_camera_channel_dense(frame, camera_to_flip, self.right_mask_ind)
        self.box_sparse.set_frame_camera_channel_dense(frame, camera_to_flip, self.left_mask_ind, right_mask)
        self.box_sparse.set_frame_camera_channel_dense(frame, camera_to_flip, self.right_mask_ind, left_mask)
        # switch confidence scores
        left_conf_scores = self.conf_preds[frame, camera_to_flip, self.left_inds]
        right_conf_scores = self.conf_preds[frame, camera_to_flip, self.right_inds]
        self.conf_preds[frame, camera_to_flip, self.left_inds] = left_conf_scores
        self.conf_preds[frame, camera_to_flip, self.right_inds] = right_conf_scores

    def align_time_channels(self):
        all_shifts = np.zeros((self.num_frames, self.num_cams, 2, 2))
        all_shifts_smoothed = np.zeros((self.num_frames, self.num_cams, 2, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                # present = self.box[frame, cam, :, :, 1]
                present = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=1)
                cm_present = self.get_fly_cm(present)
                for i, time_channel in enumerate([0, 2]):
                    # fly = self.box[frame, cam, :, :, time_channel]
                    fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=time_channel)
                    CM = self.get_fly_cm(fly)
                    shift_to_do = cm_present - CM
                    all_shifts[frame, cam, i, :] = shift_to_do

        # do shiftes
        for cam in range(self.num_cams):
            for time_channel in range(all_shifts.shape[2]):
                for axis in range(all_shifts.shape[3]):
                    vals = all_shifts[:, cam, time_channel, axis]
                    A = np.arange(vals.shape[0])
                    filtered = medfilt(vals, kernel_size=11)
                    # all_shifts_smoothed[:, cam, time_channel, axis] = filtered
                    try:
                        spline = make_smoothing_spline(A, filtered, lam=10000)
                        smoothed = spline(A)
                    except:
                        smoothed = filtered
                        print(f"spline failed in cam {cam} time channel {time_channel} and axis {axis}", flush=True)
                    all_shifts_smoothed[:, cam, time_channel, axis] = smoothed
                    pass

        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for i, time_channel in enumerate([0, 2]):
                    # fly = self.box[frame, cam, :, :, time_channel]
                    fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=time_channel)
                    shift_to_do = all_shifts_smoothed[frame, cam, i, :]
                    shifted_fly = shift(fly, shift_to_do, order=2)
                    # self.box[frame, cam, :, :, time_channel] = shifted_fly
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, time_channel, shifted_fly)

    @staticmethod
    def get_fly_cm(im_orig):
        im = gaussian_filter(im_orig, sigma=2)
        im[im < 0.8] = 0
        # im = binary_opening(im, iterations=1)
        CM = center_of_mass(im)
        return np.array(CM)

    def preprocess_masks(self):
        if self.points_to_predict == WINGS or self.points_to_predict == WINGS_AND_BODY:
            if not self.masks_flag:
                self.add_masks()
            self.adjust_masks_size()
            if self.is_video:
                self.fix_masks()

    def get_run_name(self):
        box_path_file = os.path.basename(self.box_path)
        name, ext = os.path.splitext(box_path_file)
        run_name = f"{name}_{self.model_type}_{date.today().strftime('%b %d')}"
        return run_name

    def create_2D_3D_config(self):
        json_path = self.json_2D_3D_path
        with open(json_path, "r") as jsonFile:
            data = json.load(jsonFile)

        # Change the values of some variables
        data["2D predictions path"] = self.out_path_h5
        data["align right left"] = 1

        new_json_path = os.path.join(self.run_path, "2D_to_3D_config.json")
        # Save the JSON string to a different file
        with open(new_json_path, "w") as jsonFile:
            json.dump(data, jsonFile)
        return new_json_path

    def create_run_folders(self):
        """ Creates subfolders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_path, self.run_name)

        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path):  # and not is_empty_run(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

        if os.path.exists(run_path):
            shutil.rmtree(run_path)

        os.makedirs(run_path)
        print("Created folder:", run_path, flush=True)

        return run_path

    def save_predictions_to_h5(self):
        """ save the predictions and the box of train_images to h5 file"""
        if self.num_pass > 0:
            name = "predicted_points_and_box_reprojected.h5"
        else:
            name = "predicted_points_and_box.h5"
        self.out_path_h5 = os.path.join(self.run_path, name)
        with open(f"{self.run_path}/configuration.json", 'w') as file:
            json.dump(self.config, file, indent=4)
        with h5py.File(self.out_path_h5, "w") as f:
            f.attrs["num_frames"] = self.box_sparse.shape[0]
            f.attrs["img_size"] = self.im_size
            f.attrs["box_path"] = self.box_path
            f.attrs["box_dset"] = "/box"
            f.attrs["pose_estimation_model_path"] = self.wings_pose_estimation_model_path
            f.attrs["wings_detection_model_path"] = self.wings_detection_model_path

            positions = self.predicted_points[..., :2]
            confidence_val = self.predicted_points[..., 2]

            ds_pos = f.create_dataset("positions_pred", data=positions.astype("int32"), compression="gzip",
                                      compression_opts=1)
            ds_pos.attrs["description"] = "coordinate of peak at each sample"
            ds_pos.attrs["dims"] = "(sample, joint, [x, y])"

            ds_conf = f.create_dataset("conf_pred", data=confidence_val.squeeze(), compression="gzip",
                                       compression_opts=1)
            ds_conf.attrs["description"] = "confidence map value in [0, 1.0] at peak"
            ds_conf.attrs["dims"] = "(frame, cam, joint)"
            # if self.num_frames < 2000:
            #     box = self.box_sparse.retrieve_dense_box()
            #     box[np.abs(box) < 0.001] = 0
            #     ds_conf = f.create_dataset("box", data=box, compression="gzip", compression_opts=1)
            #     ds_conf.attrs["description"] = "The predicted box and the wings1 if the wings1 were detected"
            #     ds_conf.attrs["dims"] = f"{box.shape}"

            if self.points_to_predict == WINGS or self.points_to_predict == WINGS_AND_BODY:
                ds_conf = f.create_dataset("scores", data=self.scores, compression="gzip", compression_opts=1)
                ds_conf.attrs["description"] = "the score (0->1) assigned to each wing during wing detection"
                ds_conf.attrs["dims"] = f"{self.scores.shape}"

            ds_conf = f.create_dataset("cropzone", data=self.cropzone, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "cropzone of every image for 2D to 3D projection"
            ds_conf.attrs["dims"] = f"{self.cropzone.shape}"

            ds_conf = f.create_dataset("points_3D_all", data=self.points_3D_all, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "all the points triangulations"
            ds_conf.attrs["dims"] = f"{self.points_3D_all.shape}"

            ds_conf = f.create_dataset("reprojection_errors", data=self.reprojection_errors, compression="gzip",
                                       compression_opts=1)
            ds_conf.attrs["dims"] = f"{self.reprojection_errors.shape}"
            try:
                ds_conf = f.create_dataset("triangulation_errors", data=self.triangulation_errors, compression="gzip",
                                       compression_opts=1)
            except:
                print("No triangulation errors")
            ds_conf.attrs["dims"] = f"{self.triangulation_errors.shape}"

        # save 3D points
        From2Dto3D.save_points_3D(self.run_path, self.points_3D_all, name="points_3D_all.npy")
        # points_3D = self.get_points_3D(alpha=None)
        print("predicting 3D points by checking all triangulation combinations")
        # try:
        _, self.points_3D, _, _ = Predictor2D.find_3D_points_optimize_neighbors([self.points_3D_all])
        # except Exception as e:
        #     print(f"the points 3D selection has caused an exception\n{e}")
        #     self.points_3D = self.get_points_3D(alpha=None)
        self.points_3D_smoothed = self.smooth_3D_points(self.points_3D)
        score1 = From2Dto3D.get_validation_score(self.points_3D)
        score2 = From2Dto3D.get_validation_score(self.points_3D_smoothed)
        From2Dto3D.save_points_3D(self.run_path, self.points_3D, name="points_3D.npy")
        From2Dto3D.save_points_3D(self.run_path, self.points_3D_smoothed, name="points_3D_smoothed.npy")
        readme_path = os.path.join(self.run_path, "README_scores_3D.txt")
        print(f"score1 is {score1}, score2 is {score2}", flush=True)
        with open(readme_path, "w") as f:
            # Write some text into the file
            f.write(f"The score for the points was {score1}\n")
            f.write(f"The score for the smoothed points was {score2}\n")
        # Close the file
        f.close()

    def choose_predict_method(self):
        if self.points_to_predict == WINGS:
            return self.predict_wings
        elif self.points_to_predict == BODY:
            return self.predict_body
        elif self.points_to_predict == WINGS_AND_BODY:
            if self.model_type == WINGS_AND_BODY_SAME_MODEL:
                return self.predict_wings_and_body_same_model
            if self.model_type == ALL_POINTS or self.model_type == ALL_POINTS_REPROJECTED_MASKS:
                return self.predict_all_points
            if self.model_type == ALL_CAMS_PER_WING:
                return self.predict_all_cams_per_wing
            if self.model_type == ALL_CAMS_ALL_POINTS:
                return self.predict_all_cams_all_points
            return self.predict_wings_and_body

    def predict_all_cams_per_wing(self, n=100):
        print(f"started predicting projected masks, split box into {n} parts", flush=True)
        all_points = []
        all_frames = np.arange(self.num_frames)
        n = min(self.num_frames, n)
        splited_frames = np.array_split(all_frames, n)
        for i in range(n):
            print(f"predicting part number {i + 1}")
            all_points_i = []
            for wing in range(2):
                input_wing_cams = []
                for cam_idx in range(self.num_cams):
                    input_wing_cam = self.box_sparse.get_camera_dense(camera_idx=cam_idx,
                                                                      channels=[0, 1, 2,
                                                                                self.num_times_channels + wing],
                                                                      frames=splited_frames[i])
                    input_wing_cams.append(input_wing_cam)
                input_wing = np.concatenate(input_wing_cams, axis=-1)
                output = self.wings_pose_estimation_model(input_wing)
                peaks = self.tf_find_peaks(output)
                peaks_list = [peaks[..., 0:10],
                              peaks[..., 10:20],
                              peaks[..., 20:30],
                              peaks[..., 30:40]]
                for cam in range(self.num_cams):
                    peaks_list[cam] = np.expand_dims(peaks_list[cam], axis=1)
                peaks_wing = np.concatenate(peaks_list, axis=1)
                all_points_i.append(peaks_wing)
            all_points_i = np.concatenate((all_points_i[0], all_points_i[1]), axis=-1)
            tail_points = all_points_i[..., [8, 18]]
            tail_points = np.expand_dims(np.mean(tail_points, axis=-1), axis=-1)
            head_points = all_points_i[..., [9, 19]]
            head_points = np.expand_dims(np.mean(head_points, axis=-1), axis=-1)
            wings_points = all_points_i[..., [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]]
            wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=-1)
            wings_and_body_pnts = np.transpose(wings_and_body_pnts, [0, 1, 3, 2])
            all_points.append(wings_and_body_pnts)
        all_wing_and_body_points = np.concatenate(all_points, axis=0)
        print("done predicting projected masks", flush=True)
        return all_wing_and_body_points

    def predict_all_cams_all_points(self,  n=100):
        print(f"started predicting projected masks, split box into {n} parts", flush=True)
        all_points = []
        n = max(int(self.num_frames // 50), 1)
        all_frames = np.arange(self.num_frames)
        n = min(self.num_frames, n)
        splited_frames = np.array_split(all_frames, n)
        for i in range(n):
            print(f"predicting part number {i + 1}")
            input_wing_cams = []
            for cam_idx in range(self.num_cams):
                input_wing_cam = self.box_sparse.get_camera_dense(camera_idx=cam_idx,
                                                                  channels=[0, 1, 2, 3, 4],
                                                                  frames=splited_frames[i])
                input_wing_cams.append(input_wing_cam)
            input_part = np.concatenate(input_wing_cams, axis=-1)
            output = self.wings_pose_estimation_model(input_part)
            peaks = self.tf_find_peaks(output).numpy()
            # peaks = peaks[:, :2, :]
            peaks = np.transpose(peaks, [0, 2, 1])
            peaks_per_camera = np.split(peaks, self.num_cams, axis=1)
            peaks_per_camera = np.stack(peaks_per_camera, axis=1)
            all_points.append(peaks_per_camera)
        all_wing_and_body_points = np.concatenate(all_points, axis=0)
        return all_wing_and_body_points

    def predict_all_points(self):
        all_points = []
        frames = np.arange(0, self.num_frames)
        for cam in range(self.num_cams):
            input = self.box_sparse.get_camera_dense(camera_idx=cam,
                                                                  channels=[0, 1, 2, 3, 4],
                                                                  frames=frames)
            points_cam_i, _, _, _ = self.predict_Ypk(input, self.batch_size, self.wings_pose_estimation_model)
            all_points.append(points_cam_i[np.newaxis, ...])
        wings_and_body_pnts = np.concatenate(all_points, axis=0)
        wings_and_body_pnts = np.transpose(wings_and_body_pnts, [1, 0, 3, 2])
        return wings_and_body_pnts

    def predict_wings_and_body_same_model(self):
        all_pnts = self.predict_wings()
        tail_points = all_pnts[:, :, [8, 18], :]
        tail_points = np.expand_dims(np.mean(tail_points, axis=2), axis=2)
        head_points = all_pnts[:, :, [9, 19], :]
        head_points = np.expand_dims(np.mean(head_points, axis=2), axis=2)
        wings_points = all_pnts[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], :]
        wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=2)
        return wings_and_body_pnts

    def predict_wings_and_body(self):
        wings_points = self.predict_wings()
        body_points = self.predict_body()
        wings_and_body_pnts = np.concatenate((wings_points, body_points), axis=2)
        return wings_and_body_pnts

    def predict_wings(self, n=100):
        Ypks = []
        n = max(int(self.num_frames // 50), 1)
        all_frames = np.arange(self.num_frames)
        # n = min(n, self.num_frames)
        splited_frames = np.array_split(all_frames, n)
        for cam in range(self.num_cams):
            print(f"predict camera {cam + 1}", flush=True)
            Ypks_per_wing = []
            for wing in range(2):
                # split for memory limit
                Ypk = []
                for i in range(n):
                    input_i = self.box_sparse.get_camera_dense(cam, channels=[0, 1, 2, self.num_times_channels + wing],
                                                               frames=splited_frames[i])
                    Ypk_i = self.predict_input(input_i)
                    Ypk.append(Ypk_i)
                Ypk = np.concatenate(Ypk, axis=0)
                Ypks_per_wing.append(Ypk)
            Ypk_cam = np.concatenate((Ypks_per_wing[0], Ypks_per_wing[1]), axis=-1)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        Ypk_all = np.transpose(Ypk_all, [0, 1, 3, 2])
        return Ypk_all

    def predict_input(self, input_tensor):
        if self.software == 'tensorflow':
            Ypk, _, _, _ = self.predict_Ypk(input_tensor, self.batch_size, self.wings_pose_estimation_model)
        else:
            input_tensor = input_tensor.transpose([0, 3, 1, 2])
            input_tensor = torch.from_numpy(input_tensor)
            input_tensor = input_tensor.to(self.device)
            confmaps = self.wings_pose_estimation_model(input_tensor)
            Ypk = Predictor2D.get_points_from_confmaps(confmaps)
            pass
        return Ypk

    @staticmethod
    def find_points(confmaps):
        # points = find_peaks_soft_argmax(confmaps)
        points = Predictor2D.tf_find_peaks(confmaps).numpy()
        # points = points.transpose([0, 2, 1])
        # points = points[:, :, :-1]
        return points

    @staticmethod
    def get_points_from_confmaps(confmaps):
        confmaps = confmaps.detach().cpu().numpy()
        confmaps = np.transpose(confmaps, [0, 2, 3, 1])
        output_points = Predictor2D.find_points(confmaps)
        # output_points = np.reshape(output_points, [-1, 2])
        return output_points

    def predict_body(self):
        Ypks = []
        for cam in range(self.num_cams):
            input = self.box[:, cam, :, :, :self.num_times_channels]
            Ypk_cam, _, _, _ = self.predict_Ypk(input, self.batch_size, self.head_tail_pose_estimation_model)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        Ypk_all = np.transpose(Ypk_all, [0, 1, 3, 2])
        return Ypk_all

    def add_masks(self, n=100):
        """ Add train_masks to the dataset using yolov8 segmentation model """
        all_frames = np.arange(self.num_frames)
        n = min(n, self.num_frames)
        all_frames_split = np.array_split(all_frames, n)
        for cam in range(self.num_cams):
            print(f"finds wings for camera number {cam + 1}")
            results = []
            for i in range(n):
                print(f"processing n = {i}")
                img_3_ch_i = self.box_sparse.get_camera_dense(cam, [0, 1, 2], frames=all_frames_split[i])
                img_3_ch_input = np.round(img_3_ch_i * 255)
                img_3_ch_input = [img_3_ch_input[i] for i in range(img_3_ch_input.shape[0])]
                if DETECT_WINGS_CPU:
                    with tf.device('/CPU:0'):  # Forces the operation to run on the CPU
                        results_i = self.wings_detection_model(img_3_ch_input)
                else:
                    results_i = self.wings_detection_model(img_3_ch_input)
                results.append(results_i)
            results = sum(results, [])
            for frame in range(self.num_frames):
                masks_2 = np.zeros((self.im_size, self.im_size, 2))
                result = results[frame]
                boxes = result.boxes.data.numpy()
                inds_to_keep = self.eliminate_close_vectors(boxes, 10)
                num_wings_found = np.count_nonzero(inds_to_keep)
                if num_wings_found > 0:
                    masks_found = result.masks.data.numpy()[inds_to_keep, :, :]
                else:
                    assert f"no masks found for this frame {frame} and camera {cam}"
                for wing in range(min(num_wings_found, 2)):
                    mask = masks_found[wing, :, :]
                    masks_2[:, :, wing] = mask
                self.box_sparse.set_camera_dense(camera_idx=cam, frames=[frame],
                                                 dense_camera_data=masks_2[np.newaxis, ...], channels=[3, 4])

    def adjust_masks_size(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                mask_1 = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels)
                mask_2 = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels + 1)
                mask_1 = self.adjust_mask(mask_1, self.mask_increase_initial)
                mask_2 = self.adjust_mask(mask_2, self.mask_increase_initial)
                self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels, mask_1)
                self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + 1, mask_2)

    def fix_masks(self):  # todo find out if there are even train_masks to be fixed
        """
            goes through each frame, if there is no mask for a specific wing, unite train_masks of the closest times before and after
            this frame.
            :param X: a box of size (num_frames, 20, 192, 192)
            :return: same box
            """
        search_range = 5
        problematic_masks = []
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for mask_num in range(2):
                    # mask = self.box[frame, cam, :, :, self.num_times_channels + mask_num]
                    mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam,
                                                                          self.num_times_channels + mask_num)
                    if np.all(mask == 0):  # check if all 0:
                        problematic_masks.append((frame, cam, mask_num))
                        # find previous matching mask
                        prev_mask = np.zeros(mask.shape)
                        next_mask = np.zeros(mask.shape)
                        for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                            prev_mask_i = self.box_sparse.get_frame_camera_channel_dense(prev_frame, cam,
                                                                                         self.num_times_channels + mask_num)
                            if not np.all(prev_mask_i == 0):  # there is a good mask
                                prev_mask = prev_mask_i
                                break
                        # find next matching mask
                        for next_frame in range(frame + 1, min(self.num_frames, frame + search_range)):
                            next_mask_i = self.box_sparse.get_frame_camera_channel_dense(next_frame, cam,
                                                                                         self.num_times_channels + mask_num)
                            if not np.all(next_mask_i == 0):  # there is a good mask
                                next_mask = next_mask_i
                                break
                        # combine the 2 train_masks

                        new_mask = prev_mask + next_mask
                        new_mask[new_mask >= 1] = 1

                        sz_prev_mask = np.count_nonzero(prev_mask)
                        sz_next_mask = np.count_nonzero(next_mask)
                        sz_new_mask = np.count_nonzero(new_mask)
                        if sz_prev_mask + sz_next_mask == sz_new_mask:
                            # it means that the train_masks are not overlapping
                            new_mask = prev_mask if sz_prev_mask > sz_next_mask else next_mask

                        # replace empty mask with new mask
                        self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + mask_num,
                                                                       new_mask)

    def get_cropzone(self):
        try:
            cropzone = h5py.File(self.box_path, "r")["/cropzone"][:]
        except:
            cropzone = h5py.File(self.box_path, "r")["/cropZone"][:]
        return cropzone

    def set_body_masks(self, opening_rad=6):
        """
        find the fly's body, and the distance transform for later analysis in every camera in 2D using segmentation
        """
        self.body_masks_sparse = BoxSparse(box_path=None,
                                           shape=(self.num_frames, self.num_cams, self.image_size, self.image_size, 1))
        self.body_sizes = np.zeros((self.num_frames, self.num_cams))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                # fly_3_ch = self.box[frame, cam, :, :, :self.num_time_channels]
                fly_3_ch = np.zeros((self.image_size, self.image_size, 3))
                fly_3_ch[..., 0] = self.box_sparse.get_frame_camera_channel_dense(frame, cam, 0)
                fly_3_ch[..., 1] = self.box_sparse.get_frame_camera_channel_dense(frame, cam, 1)
                fly_3_ch[..., 2] = self.box_sparse.get_frame_camera_channel_dense(frame, cam, 2)

                fly_3_ch_av = np.sum(fly_3_ch, axis=-1) / self.num_time_channels
                binary_body = fly_3_ch_av >= 0.7
                selem = disk(opening_rad)
                # Perform dilation
                dilated = dilation(binary_body, selem)
                # Perform erosion
                mask = erosion(dilated, selem)
                # body_masks[frame, cam, ...] = mask
                self.body_sizes[frame, cam] = np.count_nonzero(mask)
                self.body_masks_sparse.set_frame_camera_channel_dense(frame, cam, 0, mask)

    def get_neto_wings_masks(self):
        print("creating neto_wings_sparse", flush=True)
        self.neto_wings_sparse = BoxSparse(box_path=None,
                                           shape=(self.num_frames, self.num_cams, self.image_size, self.image_size, 2))
        print("created neto_wings_sparse", flush=True)
        self.wings_size = np.zeros((self.num_frames, self.num_cams, 2))
        print("created wings_size", flush=True)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                body_mask = self.body_masks_sparse.get_frame_camera_channel_dense(frame, cam, 0)
                fly = self.box_sparse.get_frame_camera_channel_dense(frame_idx=cam, camera_idx=cam, channel_idx=1)
                for wing_num in range(2):
                    other_wing_mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam,
                                                                                     self.num_time_channels + (
                                                                                         not wing_num))
                    wing_mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam,
                                                                               self.num_time_channels + wing_num)
                    body_and_other_wing_mask = np.bitwise_or(body_mask.astype(bool), other_wing_mask.astype(bool))
                    intersection = np.logical_and(wing_mask, body_and_other_wing_mask)
                    neto_wing = wing_mask - intersection
                    neto_wing = np.logical_and(neto_wing, fly)
                    self.neto_wings_sparse.set_frame_camera_channel_dense(frame, cam, wing_num, neto_wing)
                    self.wings_size[frame, cam, wing_num] = np.count_nonzero(neto_wing)

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        if self.num_pass == 0:
            model = YOLO(self.wings_detection_model_path)
            model.fuse()
        elif self.num_pass == 1:
            model = YOLO(self.wings_pose_estimation_model_path_second_pass)
            model.fuse()
        try:
            return model.cpu()
        except:
            return model

    @staticmethod
    def adjust_mask(mask, radius=3):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=radius).astype(int)
        return mask

    @staticmethod
    def get_pose_estimation_model_tensorflow(pose_estimation_model_path, return_model_peaks=True):
        tfconfig.enable_unsafe_deserialization()
        """ load a pretrained LEAP pose estimation model model"""
        exists = os.path.exists(pose_estimation_model_path)
        model = keras.models.load_model(pose_estimation_model_path, custom_objects={'LeakyReLU': LeakyReLU})
        if return_model_peaks:
            model = Predictor2D.convert_to_peak_outputs(model, include_confmaps=False)
        print("weights_path:", pose_estimation_model_path)
        print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))
        return model

    def get_pose_estimation_model_pytorch(self, pose_estimation_model_path):
        model = torch.jit.load(pose_estimation_model_path, map_location=torch.device(self.device))
        model.eval()
        return model

    @staticmethod
    def convert_to_peak_outputs(model, include_confmaps=False):
        """ Creates a new Keras model with a wrapper to yield channel peaks from rank-4 tensors. """
        if type(model.output) == list:
            confmaps = model.output[-1]
        else:
            confmaps = model.output

        peak_layer = Lambda(Predictor2D.tf_find_peaks, name="find_peaks_lambda")(confmaps)

        if include_confmaps:
            return keras.Model(model.input, [peak_layer, confmaps])
        else:
            return keras.Model(model.input, peak_layer)

    @staticmethod
    def tf_find_peaks(x):
        """ Finds the maximum value in each channel and returns the location and value.
        Args:
            x: rank-4 tensor (samples, height, width, channels)

        Returns:
            peaks: rank-3 tensor (samples, [x, y, val], channels)
        """

        # Store input shape
        in_shape = tf.shape(x)

        # Flatten height/width dims
        flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])

        # Find peaks in linear indices
        idx = tf.argmax(flattened, axis=1)

        # Convert linear indices to subscripts
        rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
        cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])

        # Dumb way to get actual values without indexing
        vals = tf.math.reduce_max(flattened, axis=1)

        # Return N x 3 x C tensor
        pred = tf.stack([
            tf.cast(cols, tf.float32),
            tf.cast(rows, tf.float32),
            vals],
            axis=1)
        return pred

    @staticmethod
    def predict_Ypk(X, batch_size, model_peaks, save_confmaps=False):
        """ returns a predicted dataset"""
        confmaps, confmaps_min, confmaps_max = None, None, None
        if save_confmaps:
            Ypk, confmaps = model_peaks.predict(X, batch_size=batch_size)

            # Quantize
            confmaps_min = confmaps.min()
            confmaps_max = confmaps.max()

            # Reshape
            confmaps = np.transpose(confmaps, (0, 3, 2, 1))
        else:
            Ypk = model_peaks.predict(X, batch_size=batch_size)
        return Ypk, confmaps, confmaps_min, confmaps_max

    @staticmethod
    def eliminate_close_vectors(matrix, threshold):
        # calculate pairwise Euclidean distances
        distances = cdist(matrix, matrix, 'euclidean')

        # create a mask to identify which vectors to keep
        inds_to_del = np.ones(len(matrix), dtype=bool)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if distances[i, j] < threshold:
                    # eliminate one of the vectors
                    inds_to_del[j] = False

        # return the new matrix with close vectors eliminated
        return inds_to_del

    @staticmethod
    def all_possible_combinations(lst, fraq=0.6):
        start = math.ceil(len(lst) * fraq)
        all_combinations_list = []

        for r in range(start, len(lst) + 1):
            for combo in combinations(lst, r):
                all_combinations_list.append(list(combo))

        return all_combinations_list

    @staticmethod
    def get_3D_points_median(result):
        mad = scipy.stats.median_abs_deviation(result, axis=2)
        median = np.median(result, axis=2)
        threshold = 2 * mad
        outliers_mask = np.abs(result - median[..., np.newaxis, :]) > threshold[..., np.newaxis, :]
        array_with_nan = result.copy()
        array_with_nan[outliers_mask] = np.nan
        points_3D = np.nanmedian(array_with_nan, axis=2)
        return points_3D

    @staticmethod
    def get_best_ensemble_combination(all_points_list, score_function=From2Dto3D.get_validation_score):
        models_candidates = list(range(len(all_points_list)))
        num_points_candidates = all_points_list[0].shape[2]
        points_candidates = np.arange(0, num_points_candidates)
        all_models_combinations_list = Predictor2D.all_possible_combinations(models_candidates, fraq=0.1)
        all_camera_pairs_list = Predictor2D.all_possible_combinations(points_candidates, fraq=0.01)
        best_score = float('inf')
        best_combination = None
        best_points_3D = None
        half_window = len(all_points_list[0]) // 2

        scores_dataset = []

        for combination in all_models_combinations_list:
            best_score_for_comb = float('inf')
            best_points_for_comb = None

            for cams_pairs_comb in all_camera_pairs_list:
                all_comb_points = [all_points_list[i][:, :, cams_pairs_comb, :] for i in combination]
                result = np.concatenate(all_comb_points, axis=2)
                points_3D = Predictor2D.get_3D_points_median(result)
                score = score_function(points_3D)

                if score < best_score:
                    best_score = score
                    best_combination = (combination, cams_pairs_comb)
                    best_points_3D = points_3D

                if score < best_score_for_comb:
                    best_score_for_comb = score
                    best_points_for_comb = points_3D

            scores_dataset.append({
                'model_combination': combination,
                'score': best_score_for_comb,
                'points_3D': best_points_for_comb[half_window]
            })

        return best_combination, best_points_3D, best_score, scores_dataset

    @staticmethod
    def get_window(extended_data, window_size, window_start):
        return [extended_data[i][window_start:window_start + window_size] for i in range(len(extended_data))]

    @staticmethod
    def get_extended_data(all_points_list, half_window):
        extended_data = []
        for model_output in all_points_list:
            pad_start = model_output[:half_window][::]
            pad_end = model_output[-half_window:][::]
            extended_data.append(np.concatenate([pad_start, model_output, pad_end]))
        return extended_data

    @staticmethod
    def consecutive_couples(points):
        return [[points[i], points[(i + 1) % len(points)]] for i in range(len(points))]

    @staticmethod
    def find_std_of_wings_points_dists(points):
        all_couples = Predictor2D.consecutive_couples(np.arange(points.shape[1]))
        stds = [np.std(np.linalg.norm(points[:, a, :] - points[:, b, :], axis=-1)) for a, b in all_couples]
        return np.array(stds).mean()

    @staticmethod
    def find_std_of_2_points_dists(median_points):
        dists = np.linalg.norm(median_points[:, 1, :] - median_points[:, 0, :], axis=-1)
        std = np.std(dists)
        return std

    @staticmethod
    def process_frame(args):
        window_data, score_function, frame, half_window = args
        best_combination_w, best_points_3D_w, score, scores_dataset = Predictor2D.get_best_ensemble_combination(
            window_data,
            score_function=score_function)
        chosen_point = best_points_3D_w[half_window]
        return frame, chosen_point, best_combination_w, scores_dataset

    @staticmethod
    def get_best_points_per_point_multiprocessing(all_points_list, points_inds, window_size=31,
                                                  score_function=find_std_of_2_points_dists,
                                                  candidtates_inds=(0, 1, 2, 3, 4, 5)):
        num_frames, _, num_candidates, ax = all_points_list[0].shape
        num_points = len(points_inds)
        half_window = window_size // 2
        all_chosen_points = [points[:, points_inds, ...] for points in all_points_list]
        all_chosen_points = [points[:, :, candidtates_inds, ...] for points in all_chosen_points]
        extended_data = Predictor2D.get_extended_data(all_chosen_points, half_window)

        worker_args = []
        for frame in range(num_frames):
            window_start = frame
            window_data = Predictor2D.get_window(extended_data, window_size, window_start)
            worker_args.append((window_data, score_function, frame, half_window))

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(Predictor2D.process_frame, worker_args)

        final_array = np.zeros((num_frames, num_points, 3))
        number_of_models = len(all_points_list)
        models_combinations = np.zeros((num_frames, number_of_models, num_candidates))
        all_frames_scores = []
        for frame, chosen_point, best_combination_window, scores_dataset in results:
            for i in range(len(best_combination_window[1])):
                models_combinations[frame, best_combination_window[0], best_combination_window[1][i]] = 1
            final_array[frame] = chosen_point
            all_frames_scores.append(scores_dataset)

        return final_array, models_combinations, all_frames_scores

    @staticmethod
    def find_3D_points_optimize_neighbors(all_points_list):
        left_wing_inds = list(np.arange(0, 7))
        right_wing_inds = list(np.arange(8, 15))
        head_tail_inds = [16, 17]
        side_wing_inds = [7, 15]
        num_frames = all_points_list[0].shape[0]
        final_points_3D = np.zeros((num_frames, 18, 3))

        best_left_points, model_combinations_left_points, all_frames_scores_left_points = Predictor2D.get_best_points_per_point_multiprocessing(
            all_points_list, points_inds=left_wing_inds,
            score_function=Predictor2D.find_std_of_wings_points_dists)
        score = Predictor2D.find_std_of_wings_points_dists(best_left_points)
        final_points_3D[:, left_wing_inds, :] = best_left_points
        print(f"best_left_points, score: {score}", flush=True)

        best_right_points, model_combinations_right_points, all_frames_scores_right_points = Predictor2D.get_best_points_per_point_multiprocessing(
            all_points_list, points_inds=right_wing_inds,
            score_function=Predictor2D.find_std_of_wings_points_dists)
        score = Predictor2D.find_std_of_wings_points_dists(best_right_points)
        final_points_3D[:, right_wing_inds, :] = best_right_points
        print(f"best_right_points, score: {score}", flush=True)

        best_head_tail_points, model_combinations_head_tail_points, all_frames_scores_head_tail_points = Predictor2D.get_best_points_per_point_multiprocessing(
            all_points_list, points_inds=head_tail_inds, window_size=min(73, num_frames),
            score_function=Predictor2D.find_std_of_2_points_dists)
        final_points_3D[:, head_tail_inds, :] = best_head_tail_points
        score = Predictor2D.find_std_of_2_points_dists(best_head_tail_points)
        print(f"head tail points score: {score}", flush=True)

        best_side_points, model_combinations_side_points, all_frames_scores_side_points = Predictor2D.get_best_points_per_point_multiprocessing(
            all_points_list, points_inds=side_wing_inds, window_size=min(73*3, num_frames),
            score_function=Predictor2D.find_std_of_2_points_dists)
        final_points_3D[:, side_wing_inds, :] = best_side_points
        score = Predictor2D.find_std_of_2_points_dists(best_side_points)
        print(f"side points score: {score}", flush=True)

        final_score = From2Dto3D.get_validation_score(final_points_3D)
        print(f"Final score: {final_score}", flush=True)

        all_models_combinations = np.array([model_combinations_left_points, model_combinations_right_points,
                                            model_combinations_head_tail_points, model_combinations_side_points])
        all_frames_scores = [all_frames_scores_left_points,
                             all_frames_scores_right_points,
                             all_frames_scores_head_tail_points,
                             all_frames_scores_side_points]
        return final_score, final_points_3D, all_models_combinations, all_frames_scores


if __name__ == '__main__':
    print(tf.version.VERSION)
    # points_3D_all = np.load(r"C:\Users\amita\OneDrive\Desktop\temp\points_3D_all.npy")
    # Predictor2D.find_3D_points_optimize_neighbors([points_3D_all])
    config_file = 'predict_2D_config.json'
    predictor = Predictor2D(config_file, is_maksed=True)
    predictor.run_predict_2D(save=False)



