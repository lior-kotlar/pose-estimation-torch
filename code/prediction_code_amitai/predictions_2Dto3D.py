import json
import h5py
import glob
import numpy as np
import os
import scipy.signal
from skimage.morphology import convex_hull_image
from scipy.stats import median_abs_deviation
from traingulate import Triangulate
import visualize
import matplotlib
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from validation import Validation
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d, binary_dilation
from skimage.morphology import disk, erosion, dilation
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import medfilt
from constants import *
from scipy.optimize import curve_fit

WHICH_TO_FLIP = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype(bool)
SIDE_POINTS = [7, 15]
ALPHA = 0.7


class From2Dto3D:

    def __init__(self, load_from=CONFIG, h5_file_path="", configuration_path=""):
        if load_from == CONFIG:
            with open(configuration_path) as C:
                config = json.load(C)
                self.configuration_path = configuration_path
                self.config = config
                self.points_2D_h5_path = self.config["2D predictions path"]
                self.save_path = self.config["out path"]
                self.num_cams = self.config["number of cameras"]
                self.need_left_right_alignment = bool(self.config["align right left"])
            self.preds_2D = self.load_preds_2D()
            self.cropzone = self.load_cropzone()
            self.box = self.load_box()
            self.conf_preds = self.load_conf_pred()
            self.set_attributes()
            self.triangulate = Triangulate(self.config)
            print(f"number of frames: {self.num_frames}")
            print("finding body masks")
            self.body_masks, self.body_distance_transform = self.set_body_masks()
            self.body_sizes = self.get_body_sizes()
            if self.need_left_right_alignment:
                print("started fixing right and left")
                self.fix_wings_3D_per_frame()
                print("finished fixing right and left")
            self.neto_wings_masks, self.wings_size = self.get_neto_wings_masks()
            self.all_3D_points_pairs, _, _ = self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone)

        elif load_from == H5_FILE:
            self.h5_path = h5_file_path
            self.points_2D_h5_path = self.h5_path
            self.box = self.load_box()
            self.preds_2D = self.load_preds_2D()
            self.cropzone = self.load_cropzone()
            self.conf_preds = self.load_conf_pred()
            self.set_attributes()
            f = h5py.File(self.h5_path, "r")
            self.configuration_path = f.attrs["configuration_path"]
            with open(self.configuration_path) as C:
                self.config = json.load(C)
            self.triangulate = Triangulate(self.config)
            self.num_cams = self.config["number of cameras"]
            self.body_masks = h5py.File(self.h5_path, "r")["/body_masks"][:]
            self.body_sizesbody_sizes = h5py.File(self.h5_path, "r")["/body_sizes"][:]
            self.body_distance_transform = h5py.File(self.h5_path, "r")["/body_distance_transform"][:]
            self.neto_wings_masks = h5py.File(self.h5_path, "r")["/neto_wings_masks"][:]
            self.wings_size = h5py.File(self.h5_path, "r")["/wings_size"][:]
            pass

    def set_attributes(self):
        self.image_size = self.box.shape[-2]
        self.num_frames = self.preds_2D.shape[0]
        self.num_joints = self.preds_2D.shape[2]
        self.num_time_channels = self.box.shape[-1] - 2
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.wings_pnts_inds = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.left_mask_ind = self.box.shape[-1] - 2
        self.right_mask_ind = self.box.shape[-1] - 1

    def save_data_to_h5(self):
        h5_file_name = os.path.join(self.save_path, "preprocessed_2D_to_3D.h5")
        with h5py.File(h5_file_name, "w") as f:
            f.attrs["configuration_path"] = self.configuration_path
            ds_pos = f.create_dataset("positions_pred", data=self.preds_2D, compression="gzip",
                                      compression_opts=1)
            ds_cropzone = f.create_dataset("cropzone", data=self.cropzone, compression="gzip", compression_opts=1)
            ds_box = f.create_dataset("box", data=self.box, compression="gzip", compression_opts=1)
            body_masks = f.create_dataset("body_masks", data=self.body_masks, compression="gzip", compression_opts=1)
            ds_box = f.create_dataset("body_distance_transform", data=self.body_distance_transform, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("neto_wings_masks", data=self.neto_wings_masks, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("wings_size", data=self.wings_size, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("conf_pred", data=self.conf_preds, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("body_sizes", data=self.body_sizes, compression="gzip",
                                      compression_opts=1)
        print(f"saved data to file in path:\n{h5_file_name}")

    def load_data_from_h5(self, h5_path):
        pass

    def get_points_3D(self, alpha=None):
        if alpha is None:
            scores1 = []
            scores2 = []
            alphas = [0.6, 0.7, 0.8, 0.9, 1]
            print("start choosing 3D points")
            for alpha in alphas:
                points_3D_i = self.choose_best_score_2_cams(alpha=alpha)
                smoothed_3D_i = self.smooth_3D_points(points_3D_i)
                score1 = self.get_validation_score(points_3D_i)
                score2 = self.get_validation_score(smoothed_3D_i)
                print(f"alpha = {alpha} score1 is {score1}, score2 is {score2}")
                scores1.append(score1)
                scores2.append(score2)
            min_alpha_ind = np.argmin(scores2)
            min_alpha = alphas[min_alpha_ind]
            points_3D_chosen = self.choose_best_score_2_cams(min_alpha)
            print("finish choosing 3D points")
            return points_3D_chosen, min_alpha
        else:
            points_3D_out = self.choose_best_score_2_cams(alpha=alpha)
            # smoothed_3D_i = self.smooth_3D_points(points_3D_i)
            # score1 = self.get_validation_score(points_3D_i)
            # score2 = self.get_validation_score(smoothed_3D_i)
            # print(f"alpha = {alpha} score1 is {score1}, score2 is {score2}")
            return points_3D_out

    def get_body_sizes(self):
        body_sizes = np.count_nonzero(self.body_masks, axis=(-2, -1))
        return body_sizes

    def do_smooth_3D_points(self, points_3D):
        return self.smooth_3D_points(points_3D)

    @staticmethod
    def get_validation_score(points_3D):
        return Validation.get_wings_distances_variance(points_3D)[0]

    @staticmethod
    def visualize_3D(points_3D):
        visualize.Visualizer.show_points_in_3D(points_3D)

    def visualize_2D(self, points_2D):
        visualize.Visualizer.show_predictions_all_cams(np.copy(self.box), points_2D)

    def reprojected_2D_points(self, points_3D):
        points_2D_reprojected = self.triangulate.get_reprojections(points_3D, self.cropzone)
        return points_2D_reprojected

    def get_all_3D_pnts_pairs(self, points_2D, cropzone):
        points_3D_all, reprojection_errors, triangulation_errors = \
            self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)
        return points_3D_all, reprojection_errors, triangulation_errors

    def fix_wings_3D_per_frame(self):
        """
        fix the right and left wings1 order in all the cameras according to the 3D ground truth
        one camera (camera 0) is fixed and all cameras are tested (all the options of right-left are considered)

        # step 1: make sure the right-left of chosen camera stays consistent between frames, and flip if needed
        # step 2: find which cameras needed to be flipped to minimize triangulation error, and flip them
        """
        chosen_camera = 0
        cameras_to_check = np.arange(0, 4)
        cameras_to_check = cameras_to_check[np.where(cameras_to_check != chosen_camera)]
        for frame in range(self.num_frames):
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

    def get_reprojected_points(self, smoothed=True):
        _points_3D = self.get_points_3D()
        if smoothed:
            _points_3D = self.smooth_3D_points(_points_3D)
        return self.triangulate.get_reprojections(_points_3D, self.cropzone)

    def get_reprojection_masks(self, points_3D, extend_mask_radius=4):
        points_2D_reprojected = self.triangulate.get_reprojections(points_3D, self.cropzone)
        reprojected_masks = np.zeros_like(self.neto_wings_masks)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                fly = self.box[frame, cam, :, :, 1]
                for wing in range(2):
                    points_inds = self.wings_pnts_inds[wing, :]
                    mask = np.zeros((self.image_size, self.image_size))
                    wing_pnts = np.round(points_2D_reprojected[frame, cam, points_inds, :]).astype(int)
                    mask[wing_pnts[:, 1], wing_pnts[:, 0]] = 1
                    mask = convex_hull_image(mask)  # todo switch
                    mask = binary_dilation(mask, iterations=extend_mask_radius)
                    mask = np.logical_and(mask, fly)
                    mask = binary_dilation(mask, iterations=3)
                    reprojected_masks[frame, cam, :, :, wing] = mask
        return reprojected_masks

    def get_body_points_cloud(self):
        n = 50000
        head_tail_points = self.all_3D_points_pairs[:, [-2, -1], :, :]
        self.x_body_from_cloud = np.zeros((self.num_frames, 3))
        self.x_body_from_2_points = np.zeros((self.num_frames, 3))

        self.center_of_mass_cloud = np.zeros((self.num_frames, 3))
        self.center_of_mass_2_points = np.zeros((self.num_frames, 3))

        for frame in range(self.num_frames):
            head = np.mean(head_tail_points[frame, 1, :, :], axis=0)
            tail = np.mean(head_tail_points[frame, 0, :, :], axis=0)

            center = (head + tail) / 2
            radius = np.linalg.norm(head - tail) / 2
            radius = radius * 1.5
            np.random.seed(0)
            cube = np.random.uniform(-radius, radius, size=(n, 3))
            cube = cube + center

            cropzone = np.tile(self.cropzone[frame, ...], (cube.shape[0], 1, 1))

            reprojections = np.squeeze(self.triangulate.get_reprojections(cube[:, np.newaxis, :], cropzone))
            limit = self.body_masks.shape[-2:]
            are_inside = (reprojections >= 0) & (reprojections < limit)
            are_inside_all = np.all(are_inside, axis=(1, 2))

            is_inside = [are_inside_all]
            for cam in range(self.num_cams):
                reprojections_cam = np.round(reprojections[:, cam, :]).astype(int)
                reprojections_cam[~are_inside[:, cam, :]] = 0
                mask = self.body_masks[frame, cam, :, :]
                present = self.box[frame, cam, :, :, 1]
                inside = mask[reprojections_cam[:, 1], reprojections_cam[:, 0]].astype(bool)
                is_inside.append(inside)

                # outside = np.bitwise_not(inside)
                # outide_points = reprojections_cam[outside, :]
                # inside_points = reprojections_cam[inside, :]
                # plt.imshow(present, cmap='gray')  # Display the binary mask
                # plt.scatter(inside_points[:, 0], inside_points[:, 1], color='red')  # Display the points
                # plt.scatter(outide_points[:, 0], outide_points[:, 1], color='blue')
                # plt.show()
            is_inside = np.vstack(is_inside).T
            is_inside = np.all(is_inside, axis=1)

            fly_points = cube[is_inside]

            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(fly_points)
            # The principal axis is the first principal component
            principal_axis = pca.components_[0]
            # Project points onto the principal axis
            projected_points = np.dot(fly_points - pca.mean_, principal_axis)
            # Compute distances from the projected points to the original points
            distances = np.linalg.norm(fly_points - pca.mean_ - projected_points[:, None] * principal_axis, axis=1)
            # Compute the threshold for the top 10% furthest points
            threshold = np.percentile(distances, 90)
            # Identify outliers
            outliers = distances > threshold
            fly_points = fly_points[~outliers]

            # fit again without the outliers
            # pca.fit(fly_points)
            pca = PCA(n_components=3)
            pca.fit(fly_points)

            # The principal axes are the principal components
            principal_axes = pca.components_

            # The sizes along the principal axes are the square roots of the eigenvalues
            sizes = np.sqrt(pca.explained_variance_)

            # Compute the center of mass of the points
            center_of_mass = np.mean(fly_points, axis=0)

            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Plot the points
            ax.scatter(*fly_points.T, color='orange')
            # Plot each principal axis
            for i in range(3):
                # Create a line for the i-th principal axis that goes through the center of mass
                t = np.linspace(-sizes[i], sizes[i], 100)  # parameter for the line
                line = center_of_mass + np.outer(t, principal_axes[i])

                # Plot the line
                ax.plot(*line.T)
            points = np.vstack((tail, head))
            # Plot the 2 points as red dots
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
            plt.show()
            points = np.vstack((tail, head))
            main1 = principal_axes[0]
            main2 = (head - tail) / np.linalg.norm(head - tail)
            self.x_body_from_cloud[frame, :] = main1
            self.x_body_from_2_points[frame, :] = main2
            self.center_of_mass_cloud[frame, :] = center_of_mass
            self.center_of_mass_2_points[frame, :] = np.mean(points, axis=0)

            # plt.figure()
            # ax = plt.axes(projection='3d')
            # points = np.vstack((tail, head))
            # Plot the 2 points as red dots
            # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
            # # Plot the points cloud as blue dots
            # ax.scatter(fly_points[:, 0], fly_points[:, 1], fly_points[:, 2], color='orange')
            # # Plot the cube cloud as orange dots
            # # ax.scatter(cube[~is_inside, 0], cube[~is_inside, 1], cube[~is_inside, 2], color='blue')
            # # Set the title and the labels of the axes
            # ax.set_title('2 points and points cloud in 3D')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            # # Show the plot
            # plt.show()

        axis = 1
        plt.plot(self.x_body_from_cloud[:, axis], color='blue')
        plt.plot(-self.x_body_from_2_points[:, axis], color='red')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points from the first array
        ax.scatter(*self.center_of_mass_cloud.T, color='blue')

        # Plot the points from the second array
        ax.scatter(*self.center_of_mass_2_points.T, color='red')

        plt.show()

        return 0

    def set_body_masks(self, opening_rad=6):
        """
        find the fly's body, and the distance transform for later analysis in every camera in 2D using segmentation
        """
        body_masks = np.zeros((self.num_frames, self.num_cams, self.image_size, self.image_size))
        body_distance_transform = np.zeros(body_masks.shape)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                fly_3_ch = self.box[frame, cam, :, :, :self.num_time_channels]
                fly_3_ch_av = np.sum(fly_3_ch, axis=-1) / self.num_time_channels
                binary_body = fly_3_ch_av >= 0.7
                selem = disk(opening_rad)
                # Perform dilation
                dilated = dilation(binary_body, selem)
                # Perform erosion
                mask = erosion(dilated, selem)
                distance_transform = distance_transform_edt(mask)
                body_masks[frame, cam, ...] = mask
                body_distance_transform[frame, cam, ...] = distance_transform
        return body_masks, body_distance_transform

    def get_neto_wings_masks(self):
        neto_wings = np.zeros((self.num_frames, self.num_cams, self.image_size, self.image_size, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                body_mask = self.body_masks[frame, cam, ...]
                for wing_num in range(2):
                    other_wing_mask = self.box[frame, cam, :, :, self.num_time_channels + (not wing_num)]
                    wing_mask = self.box[frame, cam, :, :, self.num_time_channels + wing_num]
                    body_and_other_wing_mask = np.bitwise_or(body_mask.astype(bool), other_wing_mask.astype(bool))
                    intersection = np.logical_and(wing_mask, body_and_other_wing_mask)
                    neto_wing = wing_mask - intersection
                    neto_wings[frame, cam, :, :, wing_num] = neto_wing
        wings_size = np.count_nonzero(neto_wings, axis=(2, 3))
        return neto_wings, wings_size

    @staticmethod
    def smooth_3D_points(points_3D, head_tail_lam=2):
        points_3D_smoothed = np.zeros_like(points_3D)
        num_joints = points_3D_smoothed.shape[1]
        for pnt in range(num_joints):
            for axis in range(3):
                # print(pnt, axis)
                vals = points_3D[:, pnt, axis]
                # set lambda as the regularising parameters: smoothing vs close to data
                # lam = 300 if pnt in SIDE_POINTS else None
                lam = None
                A = np.arange(vals.shape[0])
                if pnt in BODY_POINTS:
                    vals = medfilt(vals, kernel_size=11)
                    W = np.ones_like(vals)
                    lam = head_tail_lam * len(points_3D)
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

    @staticmethod
    def poly_model(x, *coeffs, d=3):
        y = 0
        for i in range(d + 1):
            y += coeffs[i] * x ** (d - 1)
        return y

    def choose_best_score_2_cams(self, alpha=0.7):
        """
        for each point rank the 4 different cameras by visibility, noise, size of mask, and choose the best 2
        """
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D,
                                                                                              self.cropzone)
        envelope_2D = self.get_derivative_envelope_2D()
        points_3D = np.zeros((self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            # start with head and tail points
            for ind in self.head_tail_inds:
                body_sizes = self.body_sizes[frame, :]
                candidates = points_3D_all[frame, ind, :, :]
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
                    candidates = points_3D_all[frame, pnt_num, :, :]
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

    def choose_best_reprojection_error_points(self):
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D,
                                                                                              self.cropzone)
        points_3D = np.zeros((self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for joint in range(self.num_joints):
                candidates = points_3D_all[frame, joint, ...]
                best_candidate_ind = np.argmin(reprojection_errors[frame, joint, ...])
                point_3d = candidates[best_candidate_ind]
                points_3D[frame, joint, :] = point_3d
        return points_3D

    def choose_average_points(self):
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D,
                                                                                              self.cropzone)
        points_3D = np.zeros(shape=(self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for joint in range(self.num_joints):
                candidates = points_3D_all[frame, joint, ...]
                candidates_inliers = self.find_outliers_MAD(candidates, 3)
                point_3d = candidates_inliers.mean(axis=0)
                points_3D[frame, joint, :] = point_3d
        return points_3D

    def choose_best_conf_points(self):
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D,
                                                                                              self.cropzone)
        points_3D = np.zeros(shape=(self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for joint in range(self.num_joints):
                candidates = points_3D_all[frame, joint, ...]
                confidence_scores = self.conf_preds[frame, :, joint]
                indices = np.argpartition(confidence_scores, -2)[-2:]
                indices = tuple(np.sort(indices))
                best_pair_ind = self.triangulate.all_couples.index(indices)
                best_conf_3D_point = candidates[best_pair_ind]
                points_3D[frame, joint, :] = best_conf_3D_point
        return points_3D

    @staticmethod
    def find_outliers_MAD(candidates, threshold):
        median = np.median(candidates, axis=0)
        MAD = median_abs_deviation(candidates)
        inliers = np.linalg.norm((candidates - median) / MAD, axis=-1) < threshold
        candidates_inliers = candidates[inliers]
        return candidates_inliers

    def find_which_cameras_to_flip(self, cameras_to_check, frame):
        num_of_options = len(WHICH_TO_FLIP)
        switch_scores = np.zeros(num_of_options, )
        for i, option in enumerate(WHICH_TO_FLIP):
            points_2D, cropzone = self.get_orig_2d_points_and_cropzone(frame)
            cameras_to_flip = cameras_to_check[option]
            for cam in cameras_to_flip:
                left_points = points_2D[0, cam, self.left_inds, :]
                right_points = points_2D[0, cam, self.right_inds, :]
                points_2D[0, cam, self.left_inds, :] = right_points
                points_2D[0, cam, self.right_inds, :] = left_points
            _, reprojection_errors, _ = self.get_all_3D_pnts_pairs(points_2D, cropzone)
            score = np.mean(reprojection_errors)
            switch_scores[i] = score
        cameras_to_flip = cameras_to_check[WHICH_TO_FLIP[np.argmin(switch_scores)]]
        return cameras_to_flip

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

    def flip_camera(self, camera_to_flip, frame):
        left_points = self.preds_2D[frame, camera_to_flip, self.left_inds, :]
        right_points = self.preds_2D[frame, camera_to_flip, self.right_inds, :]
        self.preds_2D[frame, camera_to_flip, self.left_inds, :] = right_points
        self.preds_2D[frame, camera_to_flip, self.right_inds, :] = left_points
        # switch train_masks in box
        self.box[frame, camera_to_flip, :, :, [self.left_mask_ind, self.right_mask_ind]] = \
            self.box[frame, camera_to_flip, :, :, [self.right_mask_ind, self.left_mask_ind]]
        # switch confidence scores
        left_conf_scores = self.conf_preds[frame, camera_to_flip, self.left_inds]
        right_conf_scores = self.conf_preds[frame, camera_to_flip, self.right_inds]
        self.conf_preds[frame, camera_to_flip, self.left_inds] = left_conf_scores
        self.conf_preds[frame, camera_to_flip, self.right_inds] = right_conf_scores

    def get_orig_2d_points_and_cropzone(self, frame):
        orig_2d_points = self.preds_2D[frame, :, np.concatenate((self.left_inds, self.right_inds)), :]
        orig_2d_points = orig_2d_points[np.newaxis, ...]
        orig_2d_points = np.transpose(orig_2d_points, [0, 2, 1, 3])

        cropzone = self.cropzone[frame]
        cropzone = cropzone[np.newaxis, ...]
        return orig_2d_points, cropzone

    def load_preds_2D(self):
        return h5py.File(self.points_2D_h5_path, "r")["/positions_pred"][:]

    def load_cropzone(self):
        return h5py.File(self.points_2D_h5_path, "r")["/cropzone"][:]

    def load_conf_pred(self):
        return h5py.File(self.points_2D_h5_path, "r")["/conf_pred"][:]

    def load_box(self):
        return h5py.File(self.points_2D_h5_path, "r")["/box"][:]

    @staticmethod
    def save_points_3D(save_dir, points_to_save, name="points_3D.npy"):
        save_path = os.path.join(save_dir, name)
        np.save(save_path, points_to_save)


def predict_3D_points_all(base_path, config_path):
    # Create an empty list to store the file paths
    file_list = []

    # Loop through the subdirectories of A
    for sub in ["mov29"]:
        # Check if the subdirectory name starts with "movie"
        dir_path = os.path.join(base_path, sub)
        dirs = glob.glob(os.path.join(dir_path, "*"))
        for dir in dirs:
            if os.path.isdir(dir):
                # Append it to the list
                predicts_file = os.path.join(dir, "predicted_points_and_box_reprojected.h5")
                if os.path.isfile(predicts_file):
                    # Append it to the list
                    file_list.append(predicts_file)
                else:
                    # Otherwise, append the file "predicted_points_and_box.h5" to the list
                    predicts_file = os.path.join(dir, "predicted_points_and_box.h5")
                    file_list.append(predicts_file)
    file_list = file_list[1:]
    for preds_file in file_list:
        print(preds_file)
        dir_path = os.path.dirname(preds_file)
        with open(config_path) as C:
            config_2D = json.load(C)
            config_2D["2D predictions path"] = preds_file
            config_2D["out path"] = dir_path
        new_config_path = os.path.join(dir_path, 'configuration predict 3D.json')
        with open(new_config_path, 'w') as file:
            json.dump(config_2D, file, indent=4)
        try:
            predictor = From2Dto3D(configuration_path=new_config_path, load_from=CONFIG)
            points_3D_all, _, _ = predictor.get_all_3D_pnts_pairs(predictor.preds_2D, predictor.cropzone)
            predictor.save_points_3D(dir_path, points_3D_all, name="points_3D_all.npy")

            points_3D, alpha = predictor.get_points_3D(alpha=None)
            smoothed_3D = predictor.smooth_3D_points(points_3D)
            predictor.save_points_3D(dir_path, points_3D, name="points_3D.npy")
            predictor.save_points_3D(dir_path, smoothed_3D, name="points_3D_smoothed.npy")

            # Open a new file called readme.txt in write mode
            readme_path = os.path.join(dir_path, "README.txt")
            score1 = predictor.get_validation_score(points_3D)
            score2 = predictor.get_validation_score(smoothed_3D)
            print(f"score1 is {score1}, score2 is {score2}")
            with open(readme_path, "w") as f:
                # Write some text into the file
                f.write(f"The score for the points was {score1}\n")
                f.write(f"The score for the smoothed points was {score2}\n")
                f.write(f"the alpha was {alpha}\n")
            # Close the file
            f.close()
        except:
            print("************** failed ****************")


def predict_3D_points_all_pairs(base_path):
    # Create an empty list to store the file paths
    all_points_file_list = []
    points_3D_file_list = []

    # Loop through the subdirectories of A
    dir_path = os.path.join(base_path)
    dirs = glob.glob(os.path.join(dir_path, "*"))
    for dir in dirs:
        if os.path.isdir(dir):
            # Append it to the list
            all_points_file = os.path.join(dir, "points_3D_all.npy")
            points_3D_file = os.path.join(dir, "points_3D.npy")
            if os.path.isfile(all_points_file):
                # Append it to the list
                all_points_file_list.append(all_points_file)
            if os.path.isfile(points_3D_file):
                points_3D_file_list.append(points_3D_file)
    all_points_arrays = []
    points_3D_arrays = []
    for array_path in all_points_file_list:
        all_points_arrays.append(np.load(array_path))

    for array_path in points_3D_file_list:
        points = np.load(array_path)
        points = points[:, :, np.newaxis, :]
        points_3D_arrays.append(points)

    big_array_all_points = np.concatenate(all_points_arrays, axis=2)
    big_array_3D_points = np.concatenate(points_3D_arrays, axis=2)

    return big_array_all_points


def find_3D_points_from_ensemble(base_path):
    big_array = predict_3D_points_all_pairs(base_path)

    present = big_array
    previous = np.roll(present, 1, axis=0)[1:-1, ...]
    next_frame = np.roll(present, -1, axis=0)[1:-1, ...]
    result = np.concatenate((present[1:-1, ...], previous, next_frame), axis=2)

    result = present

    mad = scipy.stats.median_abs_deviation(result, axis=2)
    median = np.median(result, axis=2)
    threshold = 2 * mad
    # Create a boolean mask for the outliers
    outliers_mask = np.abs(result - median[..., np.newaxis, :]) > threshold[..., np.newaxis, :]
    array_with_nan = result.copy()
    array_with_nan[outliers_mask] = np.nan
    points_3D = np.nanmedian(array_with_nan, axis=2)

    # result = present
    # points_3D = np.median(result, axis=2)

    smoothed_3D = From2Dto3D.smooth_3D_points(points_3D)

    # plt.plot(smoothed_3D[:, -1, 0], color='r', linewidth=5)
    # plt.plot(array_with_nan[:, -1, :, 0])
    # plt.plot(points_3D[:, -1, 0], color='b', linewidth=5)

    # score1 = From2Dto3D.get_validation_score(points_3D)
    # score2 = From2Dto3D.get_validation_score(smoothed_3D)

    # visualize.Visualizer.show_points_in_3D_projections(points_3D)

    score1 = From2Dto3D.get_validation_score(points_3D)
    score2 = From2Dto3D.get_validation_score(smoothed_3D)
    From2Dto3D.save_points_3D(base_path, points_3D, name="points_3D_ensemble.npy")
    From2Dto3D.save_points_3D(base_path, smoothed_3D, name="points_3D_smoothed_ensemble.npy")
    readme_path = os.path.join(base_path, "README_scores_3D_ensemble.txt")
    print(f"score1 is {score1}, score2 is {score2}")
    with open(readme_path, "w") as f:
        # Write some text into the file
        f.write(f"The score for the points was {score1}\n")
        f.write(f"The score for the smoothed points was {score2}\n")
    # Close the file
    f.close()


if __name__ == '__main__':
    config_file_path = r"2D_to_3D_config.json"
    base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies"
    predict_3D_points_all(base_path, config_file_path)

    base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies"
    for movie in ["mov29"]:
        movie_path = os.path.join(base_path, movie)
        find_3D_points_from_ensemble(movie_path)

    # path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies\mov9\movie_9_10_622_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Feb 13\predicted_points_and_box.h5"
    # box = h5py.File(path, "r")["/box"][:]
    # points_2D = h5py.File(path, "r")["/positions_pred"][:]
    # visualize.Visualizer.show_predictions_all_cams(box, points_2D)

    # path_box = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\movie_10_130_1666_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Mar 05\predicted_points_and_box_reprojected.h5"
    # box = h5py.File(path_box, "r")["/box"][:]
    # path_orig = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\movie_10_130_1666_ds_3tc_7tj.h5"
    # cropzone = h5py.File(path_orig, "r")["/cropzone"][:]
    # F = From2Dto3D(CONFIG,
    #                configuration_path=r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\2D_to_3D_config.json")
    # points_3D = np.load(
    #     r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\points_3D_smoothed_ensemble.npy")
    # reprojections_2D = F.triangulate.get_reprojections(points_3D, cropzone)
    # plt.ion()
    # print("plotting reprojections")
    # visualize.Visualizer.show_predictions_all_cams(box[:400], reprojections_2D[:400])

# compute the visibility of the point
# visibilities = np.zeros(self.num_cams,)
# for cam in range(self.num_cams):
#     body_dist_trn = self.body_distance_transform[frame, cam, :, :]
#     point = self.preds_2D[frame, cam, pnt_num, :]
#     px, py = point[0], point[1]
#     visibility = body_dist_trn[py, px]
#     visibilities[cam] = visibility
# visibilities = visibilities / np.max(visibilities)
# visibility_score = 1 - visibilities
