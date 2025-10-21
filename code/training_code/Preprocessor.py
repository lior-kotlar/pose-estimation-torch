import numpy as np
from constants import *
from utils import Config, tf_format_find_peaks
from skimage.morphology import disk, erosion, dilation
import h5py
from scipy.ndimage import binary_dilation, binary_closing

class Preprocessor:
    def __init__(self, general_configuration: Config):
        self.confmaps_orig = None
        self.box_orig = None
        self.mix_with_test = general_configuration.get_mix_with_test()
        self.mask_dilation = general_configuration.get_mask_dilation()
        # self.debug_mode = bool(config['debug mode'])
        # self.wing_size_rank = config["rank wing size"]
        # self.do_curriculum_learning = config["do curriculum learning"]
        # self.single_time_channel = bool(config["single time channel"])
        self.model_type = general_configuration.get_model_type()
        self.box, self.confmaps = self.load_dataset(general_configuration.get_data_path())

        if self.model_type == HEAD_TAIL_PER_CAM:
            self.box = self.box[:, :, :, :, :3]

        if general_configuration.get_single_time_channel():
            self.box = self.box[..., [1, -2, -1]]
        self.num_frames = self.box.shape[0]
        self.num_channels = self.box.shape[-1]
        self.num_cams = self.box.shape[1]
        self.image_size = self.box.shape[2]
        self.preprocess_function = self.get_preprocess_function()

        # get useful indexes
        self.num_dims = len(self.box.shape)
        self.num_channels = self.box.shape[-1]
        self.num_time_channels = self.num_channels - 2
        self.left_mask_ind = self.num_time_channels
        self.first_mask_ind = self.num_time_channels
        self.right_mask_ind = self.left_mask_ind + 1
        self.time_channels = np.arange(self.num_time_channels)
        self.fly_with_left_mask = np.append(self.time_channels, self.left_mask_ind)
        self.fly_with_right_mask = np.append(self.time_channels, self.right_mask_ind)

        self.num_samples = None
        if self.model_type == HEAD_TAIL_ALL_CAMS or self.model_type == HEAD_TAIL_PER_CAM:
            self.mix_with_test = False
        if general_configuration.get_debug_mode():
            if self.num_dims == 5:
                self.box = self.box[:10, :, :, :, :]
                self.confmaps = self.confmaps[:10, :, :, :, :]
            else:
                self.box = self.box[:, :10, :, :, :, :]
                self.confmaps = self.confmaps[:, :10, :, :, :, :]
            self.num_frames = self.box.shape[0]
            self.mix_with_test = False

        self.body_masks, self.body_sizes = self.get_body_masks()
        self.retrieve_points_3D(general_configuration.get_data_path())
        self.retrieve_cropzone_from_file(general_configuration.get_data_path())

    def get_box(self): 
            return self.box
    
    def get_confmaps(self):
        return self.confmaps
    
    def get_body_masks(self, opening_rad=6):
        """
        find the fly's body, and the distance transform for later analysis in every camera in 2D using segmentation
        """
        body_masks = np.zeros(shape=(self.num_frames, self.num_cams, self.image_size, self.image_size))
        body_sizes = np.zeros((self.num_frames, self.num_cams))
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
                body_sizes[frame, cam] = np.count_nonzero(mask)
                body_masks[frame, cam, ...] = mask
        return body_masks, body_sizes
    

    def retrieve_points_3D(self, data_path):
        self.points_3D = h5py.File(data_path, "r")["/points_3D"][:]
        self.points_3D = np.transpose(self.points_3D, [1, 2, 0])[:self.box.shape[0]]
        self.num_points = self.points_3D.shape[1]
        self.num_wing_points = self.num_points - 2
        self.left_inds = np.arange(0, self.num_wing_points // 2)
        self.right_inds = np.arange(self.num_wing_points // 2, self.num_wing_points)
        self.head_tail_inds = np.array([-2, -1])
        left_wing = self.points_3D[:, np.append(self.left_inds, self.head_tail_inds), :]
        right_wing = self.points_3D[:, np.append(self.right_inds, self.head_tail_inds), :]
        self.points_3D_per_wing = np.concatenate((left_wing, right_wing), axis=0)
        self.points_3D_per_camera = np.repeat(np.expand_dims(self.points_3D, axis=1), self.box.shape[1], axis=1)

    def retrieve_cropzone_from_file(self, data_path):
        self.cropzone = h5py.File(data_path, "r")["/cropZone"][:]

    def get_preprocess_function(self):
        if self.model_type == HEAD_TAIL_ALL_CAMS:
            return self.do_preprocess_HEAD_TAIL_ALL_CAMS
        elif self.model_type == HEAD_TAIL_PER_CAM or self.model_type == HEAD_TAIL_PER_CAM_POINTS_LOSS:
            return self.do_preprocess_HEAD_TAIL_PER_CAM
        elif (self.model_type == MODEL_18_POINTS_PER_WING or
              self.model_type == MODEL_18_POINTS_PICK_3_BEST_CAMERAS or
              self.model_type == MODEL_18_POINTS_3_CAMERAS_ONLY):
            return self.do_preprocess_18_pnts
        
        # if self.model_type == ALL_POINTS_MODEL or self.model_type == HEAD_TAIL or self.model_type == TWO_WINGS_TOGATHER or self.model_type == ALL_POINTS_MODEL_VIT:
        #     return self.reshape_to_cnn_input
        # elif self.model_type == PER_WING_MODEL or self.model_type == C2F_PER_WING \
        #      or self.model_type == COARSE_PER_WING \
        #         or self.model_type == VGG_PER_WING or self.model_type == HOURGLASS:
        #     return self.do_reshape_per_wing
        # elif self.model_type == TRAIN_ON_2_GOOD_CAMERAS_MODEL or self.model_type == TRAIN_ON_3_GOOD_CAMERAS_MODEL:
        #     return self.do_reshape_per_wing
        # elif self.model_type == BODY_PARTS_MODEL:
        #     return self.reshape_to_body_parts
        # elif (self.model_type == ALL_CAMS or self.model_type == ALL_CAMS_AND_3_GOOD_CAMS
        #       or self.model_type == PER_WING_SMALL_WINGS_MODEL):
        #     return self.do_reshape_per_wing
        # elif self.model_type == HEAD_TAIL_ALL_CAMS:
        #     return self.do_preprocess_HEAD_TAIL_ALL_CAMS
        # elif self.model_type == HEAD_TAIL_PER_CAM or self.model_type == HEAD_TAIL_PER_CAM_POINTS_LOSS:
        #     return self.do_preprocess_HEAD_TAIL_PER_CAM
        # elif (self.model_type == MODEL_18_POINTS_PER_WING or self.model_type == MODEL_18_POINTS_PICK_3_BEST_CAMERAS or \
        #         self.model_type == RESNET_18_POINTS_PER_WING or self.model_type == MODEL_18_POINTS_PER_WING_VIT or
        #       self.model_type == MODEL_18_POINTS_PER_WING_VIT_TO_POINTS or
        #       self.model_type == MODEL_18_POINTS_3_GOOD_CAMERAS_VIT or self.model_type == MODEL_18_POINTS_3_CAMERAS_ONLY):
        #     return self.do_preprocess_18_pnts
        # elif self.model_type == ALL_CAMS_ALL_POINTS:
        #     return self.reshape_to_all_cams_all_points
        # elif (self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_VIT
        #       or self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_CNN or self.model_type == ALL_CAMS_VIT):
        #     return self.reshape_for_ALL_CAMS_18_POINTS

    
    def do_preprocess_HEAD_TAIL_ALL_CAMS(self):
        # self.box = self.box[..., :3]
        self.box = np.concatenate((self.box[0, ...],
                                   self.box[1, ...]))
        self.box = np.concatenate((self.box[:, 0, ...],
                                   self.box[:, 1, ...],
                                   self.box[:, 2, ...],
                                   self.box[:, 3, ...]), axis=-1)
        self.confmaps = np.concatenate((self.confmaps[0, ...],
                                        self.confmaps[1, ...]))
        self.confmaps = np.concatenate((self.confmaps[:, 0, ...],
                                        self.confmaps[:, 1, ...],
                                        self.confmaps[:, 2, ...],
                                        self.confmaps[:, 3, ...]), axis=-1)
        self.num_samples = self.box.shape[0]

    
    def do_preprocess_HEAD_TAIL_PER_CAM(self):
        self.box = self.box[..., :3]
        self.box = np.concatenate((self.box[0, ...],
                                   self.box[1, ...]))
        self.box = np.concatenate((self.box[:, 0, ...],
                                   self.box[:, 1, ...],
                                   self.box[:, 2, ...],
                                   self.box[:, 3, ...]), axis=0)
        self.confmaps = np.concatenate((self.confmaps[0, ...],
                                        self.confmaps[1, ...]))
        self.confmaps = np.concatenate((self.confmaps[:, 0, ...],
                                        self.confmaps[:, 1, ...],
                                        self.confmaps[:, 2, ...],
                                        self.confmaps[:, 3, ...]), axis=0)
        self.num_samples = self.box.shape[0]

    def do_preprocess_18_pnts(self):
        head_tail_confmaps = self.confmaps[..., -2:]
        num_of_frames = head_tail_confmaps.shape[0]
        wings_confmaps = self.confmaps[..., :-2]
        self.box, wings_confmaps = self.split_per_wing(self.box, wings_confmaps, PER_WING_MODEL, RANDOM_TRAIN_SET)
        left_confmaps = wings_confmaps[:num_of_frames]
        right_confmaps = wings_confmaps[num_of_frames:]
        left_confmaps = np.concatenate((left_confmaps, head_tail_confmaps), axis=-1)
        right_confmaps = np.concatenate((right_confmaps, head_tail_confmaps), axis=-1)
        self.confmaps = np.concatenate((left_confmaps, right_confmaps), axis=0)
        self.adjust_masks_size_per_wing()

        self.wings_sizes = self.get_neto_wings_masks()
        wings_sizes_left = self.wings_sizes[..., 0]
        wings_sizes_right = self.wings_sizes[..., 1]
        wings_sizes_all = np.concatenate((wings_sizes_left, wings_sizes_right), axis=0)

        if self.model_type == MODEL_18_POINTS_PICK_3_BEST_CAMERAS:
            self.box, self.confmaps, _, _, _ = self.take_n_good_cameras(self.box, self.confmaps, wings_sizes_all, 3)
        elif self.model_type == MODEL_18_POINTS_3_CAMERAS_ONLY and self.box.shape[1] == 4:
            self.box, self.confmaps, _, _, _ = self.remove_bottom_camera(self.box, self.confmaps)
        self.box = np.reshape(self.box, shape=[self.box.shape[0] * self.box.shape[1],
                                                  self.box.shape[2], self.box.shape[3],
                                                  self.box.shape[4]])
        self.confmaps = np.reshape(self.confmaps,
                                   shape=[self.confmaps.shape[0] * self.confmaps.shape[1],
                                             self.confmaps.shape[2], self.confmaps.shape[3],
                                             self.confmaps.shape[4]])
        self.box = self.box.transpose(0, 3, 1, 2)
        self.confmaps = self.confmaps.transpose(0, 3, 1, 2)
        self.num_samples = self.box.shape[0]

    def load_dataset(self, data_path):
        """ Loads and normalizes datasets. """
        # Load
        X_dset = "box"
        Y_dset = "confmaps"
        with h5py.File(data_path, "r") as f:
            X = f[X_dset][:]
            Y = f[Y_dset][:]

        # Adjust dimensions
        X = self.preprocess(X, permute=None)
        Y = self.preprocess(Y, permute=None)
        if X.shape[0] != 2 and X.shape[1] != 4:
            X = X.T
        if Y.shape[0] != 2 or Y.shape[1] == 192:
            Y = Y.T
        return X, Y
    
    def preprocess(self, X, permute=(0, 3, 2, 1)):

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
    

    def do_preprocess(self):
        # if self.mix_with_test:
        #     self.do_mix_with_test()
        self.preprocess_function()

    
    def split_per_wing(self, box, confmaps, model_type, trainset_type):
        """ make sure the confmaps fits the wings1 """
        min_in_mask = 3
        num_joints = confmaps.shape[-1]
        num_joints_per_wing = int(num_joints / 2)
        LEFT_INDEXES = np.arange(0, num_joints_per_wing)
        RIGHT_INDEXES = np.arange(num_joints_per_wing, 2 * num_joints_per_wing)

        left_wing_box = box[:, :, :, :, self.fly_with_left_mask]
        right_wing_box = box[:, :, :, :, self.fly_with_right_mask]
        right_wing_confmaps = confmaps[:, :, :, :, LEFT_INDEXES]
        left_wing_confmaps = confmaps[:, :, :, :, RIGHT_INDEXES]

        num_frames = box.shape[0]
        num_cams = box.shape[1]
        num_pts_per_wing = right_wing_confmaps.shape[-1]
        left_peaks = np.zeros((num_frames, num_cams, 2, num_pts_per_wing))
        right_peaks = np.zeros((num_frames, num_cams, 2, num_pts_per_wing))
        for cam in range(num_cams):
            l_p = tf_format_find_peaks(left_wing_confmaps[:, cam, :, :, :])[:, :2, :]
            r_p = tf_format_find_peaks(right_wing_confmaps[:, cam, :, :, :])[:, :2, :]
            left_peaks[:, cam, :, :] = l_p
            right_peaks[:, cam, :, :] = r_p

        left_peaks = left_peaks.astype(int)
        right_peaks = right_peaks.astype(int)

        new_left_wing_box = np.zeros(left_wing_box.shape)
        new_right_wing_box = np.zeros(right_wing_box.shape)
        new_right_wing_confmaps = np.zeros(right_wing_confmaps.shape)
        new_left_wing_confmaps = np.zeros(left_wing_confmaps.shape)

        num_of_bad_masks = 0
        # fit confmaps to wings1
        num_frames = box.shape[0]
        for frame in range(num_frames):
            for cam in range(num_cams):
                append = True
                fly_image = left_wing_box[frame, cam, :, :, self.time_channels]

                left_confmap = left_wing_confmaps[frame, cam, :, :, :]
                right_confmap = right_wing_confmaps[frame, cam, :, :, :]

                left_mask = left_wing_box[frame, cam, :, :, self.first_mask_ind]
                right_mask = right_wing_box[frame, cam, :, :, self.first_mask_ind]

                left_peaks_i = left_peaks[frame, cam, :, :]
                right_peaks_i = right_peaks[frame, cam, :, :]

                # check peaks
                left_values = 0
                right_values = 0
                for i in range(left_peaks_i.shape[-1]):
                    left_values += left_mask[left_peaks_i[1, i], left_peaks_i[0, i]]
                    right_values += right_mask[right_peaks_i[1, i], right_peaks_i[0, i]]

                # switch train_masks if peaks are completely missed
                if left_values < min_in_mask and right_values < min_in_mask:
                    temp = left_mask
                    left_mask = right_mask
                    right_mask = temp

                # check peaks again
                left_values = 0
                right_values = 0
                for i in range(left_peaks_i.shape[-1]):
                    left_values += left_mask[left_peaks_i[1, i], left_peaks_i[0, i]]
                    right_values += right_mask[right_peaks_i[1, i], right_peaks_i[0, i]]

                # don't append if one mask is missing # later fix: all masks exist
                mask_exist = True
                # if left_values < min_in_mask or right_values < min_in_mask:
                #     mask_exist = False
                #     num_of_bad_masks += 1

                if trainset_type == MOVIE_TRAIN_SET or (trainset_type == RANDOM_TRAIN_SET and mask_exist):
                    # copy fly image
                    new_left_wing_box[frame, cam, :, :, self.time_channels] = fly_image
                    new_left_wing_box[frame, cam, :, :, self.first_mask_ind] = left_mask
                    # copy mask
                    new_right_wing_box[frame, cam, :, :, self.time_channels] = fly_image
                    new_right_wing_box[frame, cam, :, :, self.first_mask_ind] = right_mask
                    # copy confmaps
                    new_right_wing_confmaps[frame, cam, :, :, :] = right_confmap
                    new_left_wing_confmaps[frame, cam, :, :, :] = left_confmap

        # save the original box and confidence maps
        self.box_orig = np.zeros(list(new_left_wing_box.shape[:-1]) + [5])
        self.box_orig[..., [0, 1, 2, 3]] = new_left_wing_box
        self.box_orig[..., -1] = new_right_wing_box[..., -1]
        self.confmaps_orig = np.concatenate((new_left_wing_confmaps, new_right_wing_confmaps), axis=-1)

        if model_type == PER_WING_MODEL:
            box = np.concatenate((new_left_wing_box, new_right_wing_box), axis=0)
            confmaps = np.concatenate((new_left_wing_confmaps, new_right_wing_confmaps), axis=0)

        elif model_type == ALL_POINTS_MODEL:
            # copy fly
            box[:, :, :, :, self.time_channels] = new_left_wing_box[:, :, :, :, self.time_channels]
            # copy left mask
            box[:, :, :, :, self.left_mask_ind] = new_left_wing_box[:, :, :, :, self.first_mask_ind]
            box[:, :, :, :, self.right_mask_ind] = new_right_wing_box[:, :, :, :, self.first_mask_ind]
            confmaps[:, :, :, :, LEFT_INDEXES] = new_left_wing_confmaps
            confmaps[:, :, :, :, RIGHT_INDEXES] = new_right_wing_confmaps

        print(f"finish preprocess. number of bad train_masks = {num_of_bad_masks}")
        return box, confmaps
    
    
    def adjust_masks_size_per_wing(self):
        num_frames = self.box.shape[0]
        num_cams = self.box.shape[1]
        for frame in range(num_frames):
            for cam in range(num_cams):
                mask = self.box[frame, cam, :, :, self.first_mask_ind]
                adjusted_mask = self.adjust_mask(mask)
                self.box[frame, cam, :, :, self.first_mask_ind] = adjusted_mask

    def adjust_mask(self, mask):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=self.mask_dilation).astype(int)
        return mask
    
    def get_neto_wings_masks(self):
        wings_size = np.zeros((self.num_frames, self.num_cams, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                body_mask = self.body_masks[frame, cam, :, :]
                fly = self.box_orig[frame, cam, :, :, 1]
                for wing_num in range(2):
                    other_wing_mask = self.box_orig[frame, cam, :, :, self.num_time_channels + (not wing_num)]
                    wing_mask = self.box_orig[frame, cam, :, :, self.num_time_channels + wing_num]
                    body_and_other_wing_mask = np.bitwise_or(body_mask.astype(bool), other_wing_mask.astype(bool))
                    intersection = np.logical_and(wing_mask, body_and_other_wing_mask)
                    neto_wing = wing_mask - intersection
                    neto_wing = np.logical_and(neto_wing, fly)
                    wings_size[frame, cam, wing_num] = np.count_nonzero(neto_wing)
        return wings_size
    
    def take_n_good_cameras(box, confmaps, all_wings_sizes, n, wing_size_rank=3):
        num_frames = box.shape[0]
        num_cams = box.shape[1]
        new_num_cams = n
        image_shape = box.shape[2]
        num_channels_box = box.shape[-1]
        num_channels_confmap = confmaps.shape[-1]
        new_box = np.zeros((num_frames, new_num_cams, image_shape, image_shape, num_channels_box))
        new_confmap = np.zeros((num_frames, new_num_cams, image_shape, image_shape, num_channels_confmap))
        small_wings_box = np.zeros((num_frames, image_shape, image_shape, num_channels_box))
        small_wings_confmaps = np.zeros((num_frames, image_shape, image_shape, num_channels_confmap))
        d_size_wings_inds = np.zeros((num_frames,))
        for frame in range(num_frames):
            wings_size = all_wings_sizes[frame]
            wings_size_argsort = np.argsort(wings_size)[::-1]
            d_size_wing_ind = wings_size_argsort[wing_size_rank]
            d_size_wings_inds[frame] = d_size_wing_ind
            best_n_cameras = np.sort(wings_size_argsort[:new_num_cams])
            new_box[frame, ...] = box[frame, best_n_cameras, ...]
            new_confmap[frame, ...] = confmaps[frame, best_n_cameras, ...]
            small_wings_box[frame, ...] = box[frame, d_size_wing_ind, ...]
            small_wings_confmaps[frame, ...] = confmaps[frame, d_size_wing_ind, ...]
        return new_box, new_confmap, small_wings_box, small_wings_confmaps, d_size_wings_inds.astype(int)
    
    def remove_bottom_camera(box, confmaps, bottom_camera_ind=0):
        new_box = np.delete(box, bottom_camera_ind, axis=1)
        new_confmaps = np.delete(confmaps, bottom_camera_ind, axis=1)
        return new_box, new_confmaps
