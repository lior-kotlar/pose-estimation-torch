import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from visualize import Visualizer
from scipy.signal import savgol_filter, find_peaks
from numpy.polynomial.polynomial import Polynomial
from utils import get_start_frame
from utils import find_flip_in_files
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.widgets import Slider
import re
from scipy.spatial.distance import mahalanobis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from visualize import create_gif_one_movie
from validation import Validation
# matplotlib.use('TkAgg')


SAMPLING_RATE = 16000
dt = 1 / 16000
LEFT = 0
RIGHT = 1
NUM_TIPS_EACH_SIZE_Y_BODY = 10
WINGS_JOINTS_INDS = [7, 15]
WING_TIP_IND = 2
UPPER_PLANE_POINTS = [0, 1, 2, 6]
LOWER_PLANE_POINTS = [2, 3, 4, 5, 6]


class HalfWingbit:
    def __init__(self, start, end, phi_vals, frames, start_peak_val, end_peak_val, average_value, strock_direction):
        self.start = start
        self.end = end
        self.middle_frame = (end - start) / 2
        self.frames = frames
        self.phi_vals = phi_vals
        self.start_peak_val = start_peak_val
        self.end_peak_val = end_peak_val
        self.average_value = average_value
        self.strock_direction = strock_direction
        self.hign_peak_val = max(self.start_peak_val, self.end_peak_val)
        self.low_peak_val = min(self.start_peak_val, self.end_peak_val)
        self.amplitude = self.hign_peak_val - self.low_peak_val


class FullWingBit:
    def __init__(self, start, end, frames, phi_vals, theta_vals, psi_vals):
        self.start = start
        self.end = end
        self.middle_frame = (end - start) / 2
        self.frames = frames
        self.phi_vals = phi_vals
        self.theta_vals = theta_vals,
        self.psi_vals = psi_vals


class FullWingBitBody:
    def __init__(self, start, end, frames,
                 torque,avarage_torque_body,
                 points_3D, CM_dot, CM_speed,
                 yaw_angle, pitch_angle, roll_angle,
                 roll_dot, pitch_dot, yaw_dot,
                 roll_dot_dot, yaw_dot_dot, pitch_dot_dot,
                 omega_body, angular_speed_body,
                 omega_body_dot, angular_acceleration_body,
                 p, q, r, left_amplitude, right_amplitude,
                 left_phi_max, left_phi_min, right_phi_max,
                 right_phi_min, phi_max_average, phi_min_average,
                 phi_mid_frame_left, phi_mid_frame_right, phi_mid_frame_average,
                 left_theta_mid_frame_dif,
                 right_theta_mid_frame_dif,
                 avarage_theta_mid_frame_dif,
                 left_psi_mid_downstroke,
                 left_psi_mid_upstroke,
                 left_psi_diff,
                 right_psi_mid_downstroke,
                 right_psi_mid_upstroke,
                 right_psi_diff,
                 average_psi_mid_downstroke,
                 average_psi_mid_upstroke,
                 average_psi_diff,
                 left_minus_right_psi_mid_downstroke,
                 left_minus_right_psi_mid_upstroke,
                 phi_frontstroke_left_minus_right,
                 phi_upstroke_avarage,
                 wings_phi_left_dot,
                 wings_phi_right_dot,
                 wings_theta_left_dot,
                 wings_theta_right_dot,
                 wings_psi_left_dot,
                 wings_psi_right_dot
                 ):
        self.start = start
        self.end = end
        self.middle_frame = (end - start) / 2
        self.frames = frames
        self.points_3D = points_3D
        self.CM_dot = CM_dot
        self.CM_speed = CM_speed
        self.yaw_angle = yaw_angle
        self.pitch_angle = pitch_angle
        self.roll_angle = roll_angle
        self.roll_dot = roll_dot
        self.pitch_dot = pitch_dot
        self.yaw_dot = yaw_dot
        self.roll_dot_dot = roll_dot_dot
        self.pitch_dot_dot = pitch_dot_dot
        self.yaw_dot_dot = yaw_dot_dot
        self.omega_body_dot = omega_body_dot
        self.angular_acceleration_body = angular_acceleration_body
        self.omega_body = omega_body
        self.wx, self.wy, self.wz = self.omega_body.T
        self.wx_dot, self.wy_dot, self.wz_dot = self.omega_body_dot.T
        self.angular_speed_body = angular_speed_body
        self.p = p
        self.q = q
        self.r = r
        self.torque = torque

        # wings dot attributes
        self.wings_phi_left_dot = wings_phi_left_dot
        self.wings_phi_right_dot = wings_phi_right_dot
        self.wings_theta_left_dot = wings_theta_left_dot
        self.wings_theta_right_dot = wings_theta_right_dot
        self.wings_psi_left_dot = wings_psi_left_dot
        self.wings_psi_right_dot = wings_psi_right_dot

        # phi attributes
        self.phi_amplitude_left_take = left_amplitude
        self.phi_amplitude_right_take = right_amplitude
        self.phi_amplitude_left_minus_right_take = left_amplitude - right_amplitude
        self.phi_max_left_take = left_phi_max
        self.phi_min_left_take = left_phi_min
        self.phi_max_right_take = right_phi_max
        self.phi_min_right_take = right_phi_min
        self.phi_max_average_take = phi_max_average
        self.phi_min_average_take = phi_min_average  # mean front stoke angle per wingbeat
        self.phi_mid_frame_left_take = phi_mid_frame_left
        self.phi_mid_frame_right_take = phi_mid_frame_right
        self.phi_mid_frame_average_take = phi_mid_frame_average
        self.phi_frontstroke_left_minus_right_take = phi_frontstroke_left_minus_right
        self.phi_upstroke_avarage_take = phi_upstroke_avarage

        # theta attribtes
        self.theta_mid_frame_dif_left_take = left_theta_mid_frame_dif
        self.theta_mid_frame_dif_right_take = right_theta_mid_frame_dif
        self.theta_mid_frame_dif_avarage_take = avarage_theta_mid_frame_dif

        # psi attributes
        self.psi_mid_downstroke_left_take = left_psi_mid_downstroke
        self.psi_mid_upstroke_left_take = left_psi_mid_upstroke
        self.psi_diff_left_take = left_psi_diff
        self.psi_mid_downstroke_right_take = right_psi_mid_downstroke
        self.psi_mid_upstroke_right_take = right_psi_mid_upstroke
        self.psi_diff_right_take = right_psi_diff
        self.psi_mid_downstroke_average_take = average_psi_mid_downstroke
        self.psi_mid_upstroke_average_take = average_psi_mid_upstroke
        self.psi_diff_average_take = average_psi_diff
        self.psi_mid_downstroke_left_minus_right_take = left_minus_right_psi_mid_downstroke
        self.psi_mid_upstroke_left_minus_right_take = left_minus_right_psi_mid_upstroke

        # body attributes
        self.torque_body_x_take = avarage_torque_body[0]
        self.torque_body_y_take = avarage_torque_body[1]
        self.torque_body_z_take = avarage_torque_body[2]
        self.body_CM_speed_take = np.mean(self.CM_speed)
        self.body_yaw_angle_take = np.mean(self.yaw_angle)
        self.body_pitch_angle_take = np.mean(self.pitch_angle)
        self.body_roll_angle_take = np.mean(self.roll_angle)
        self.body_roll_dot_take = np.mean(self.roll_dot)
        self.body_pitch_dot_take = np.mean(self.pitch_dot)
        self.body_yaw_dot_take = np.mean(self.yaw_dot)
        self.body_roll_dot_dot_take = np.mean(self.roll_dot_dot)
        self.body_pitch_dot_dot_take = np.mean(self.pitch_dot_dot)
        self.body_yaw_dot_dot_take = np.mean(self.yaw_dot_dot)
        self.body_angular_speed_take = np.mean(self.angular_speed_body)
        self.body_angular_acceleration_take = np.mean(self.angular_acceleration_body)
        self.body_wx_take, self.body_wy_take, self.body_wz_take = self.wx.mean(), self.wy.mean(), self.wz.mean()
        self.body_wx_dot_take, self.body_wy_dot_take, self.body_wz_dot_take = self.wx_dot.mean(), self.wy_dot.mean(), self.wz_dot.mean()

        # doesnt take
        self.body_CM_dot_mean = np.mean(self.CM_dot, axis=0)
        self.body_p = np.mean(self.p)
        self.body_q = np.mean(self.q)
        self.body_r = np.mean(self.r)
        self.body_omega_body = np.mean(self.omega_body, axis=0)


class FlightAnalysis:
    def __init__(self, points_3D_path="", find_auto_correlation=True,
                       create_html=False, create_mp4=False, create_h5=False,
                       show_phi=False, validation=False, points_3D=None, smooth=True):
        if not validation:
            self.points_3D_path = points_3D_path
            self.dir = os.path.dirname(self.points_3D_path)
            self.points_3D = self.load_points()
            self.first_analysed_frame = self.get_first_analysed_frame()
            self.cut = self.find_cut_value()

            if self.cut:
                cut_frame = self.cut - self.first_analysed_frame
                self.points_3D = self.points_3D[:cut_frame]
        else:
            assert type(points_3D) == np.ndarray, "point_3D should contain an array of points"
            self.points_3D = points_3D
            self.first_analysed_frame = 0

        # set attributes
        self.num_joints = self.points_3D.shape[1]
        self.num_frames = self.points_3D.shape[0]
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.left_wing_inds = self.left_inds[:-1]
        self.right_wing_inds = self.right_inds[:-1]
        self.all_wing_inds = np.array([self.left_wing_inds, self.right_wing_inds])
        self.all_non_body_points = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.wings_tips_inds = [WING_TIP_IND, WING_TIP_IND + self.num_points_per_wing]

        # calculate things
        self.points_3D = FlightAnalysis.enforce_3D_consistency(self.points_3D, self.right_inds, self.left_inds)
        # Visualizer.show_points_in_3D(self.points_3D)
        self.head_tail_points = self.get_head_tail_points(smooth=smooth)
        self.points_3D[:, self.head_tail_inds, :] = self.head_tail_points
        self.x_body = self.get_head_tail_vec()
        if not validation:
            self.set_right_left()
        self.wings_tips_left, self.wings_tips_right = self.get_wing_tips()
        self.wings_joints_points = self.get_wings_joints(smooth=smooth)
        self.wings_joints_vec = self.get_wings_joints_vec()
        self.all_2_planes = self.get_planes(points_inds=self.all_wing_inds)
        self.all_upper_planes = self.get_planes(points_inds=self.all_wing_inds[:, UPPER_PLANE_POINTS])
        self.all_lower_planes = self.get_planes(points_inds=self.all_wing_inds[:, LOWER_PLANE_POINTS])

        # visualize planes
        # Visualizer.show_points_and_wing_planes_3D(self.points_3D[:100], self.all_upper_planes)

        self.left_wing_CM, self.right_wing_CM = self.get_wings_CM()
        self.left_wing_span, self.right_wing_span = self.get_wings_spans()
        self.left_wing_chord, self.right_wing_chord = self.get_wings_cords()
        self.left_wing_lower_chord, self.right_wing_lower_chord = self.get_wings_lower_cords()
        self.left_deformation_angle, self.right_deformation_angle  = self.calculate_deformation_angle()

        # check = FlightAnalysis.row_wize_dot(self.left_wing_chord, self.left_wing_chord)
        self.center_of_mass = self.get_center_of_mass()
        self.CM_speed, self.CM_dot = self.get_body_speed()

        self.y_body, self.first_y_body_frame, self.end_frame = self.get_roni_y_body()
        self.z_body = self.get_z_body()

        # Visualizer.visualize_rotating_frames(self.x_body,
        #                                      self.y_body,
        #                                      self.z_body,
        #                                      omega=None)

        self.yaw_angle_roni = self.get_body_yaw(self.x_body)
        self.pitch_angle_roni = self.get_body_pitch(self.x_body)
        self.roll_angle_roni = self.get_body_roll(phi=self.yaw_angle_roni,
                                             theta=self.pitch_angle_roni,
                                             x_body=self.x_body,
                                             yaw=self.yaw_angle_roni,
                                             pitch=self.pitch_angle_roni,
                                             start=self.first_y_body_frame,
                                             end=self.end_frame,
                                             y_body=self.y_body)

        self.yaw_angle, self.pitch_angle, self.roll_angle = self.get_yaw_pitch_roll_from_euler()

        self.yaw_dot = self.get_dot(self.yaw_angle, sampling_rate=SAMPLING_RATE)
        self.pitch_dot = self.get_dot(self.pitch_angle, sampling_rate=SAMPLING_RATE)
        self.roll_dot = self.get_roll_dot(self.roll_angle)

        self.yaw_dot_dot = self.get_dot(self.yaw_dot, sampling_rate=SAMPLING_RATE)
        self.pitch_dot_dot = self.get_dot(self.pitch_dot, sampling_rate=SAMPLING_RATE)
        self.roll_dot_dot = self.get_roll_dot(self.roll_dot)
        self.p, self.q, self.r = self.get_pqr()


        self.average_roll_angle = np.mean(self.roll_angle[self.first_y_body_frame:self.end_frame])
        self.average_roll_speed = np.nanmean(np.abs(self.roll_dot))
        self.average_roll_velocity = np.nanmean(self.roll_dot)
        self.stroke_planes = self.get_stroke_planes()
        self.wing_tips_speed = self.get_wing_tips_speed()

        self.wings_phi_left, self.wings_phi_right = self.get_wings_phi()
        self.wings_theta_left, self.wings_theta_right = self.get_wings_theta()
        self.wings_psi_left, self.wings_psi_right = self.get_wings_psi()

        # plt.plot(self.left_deformation_angle)
        # plt.plot(self.wings_phi_left)
        # # plt.plot(self.wings_phi_left, self.left_deformation_angle, '-')
        # plt.show()
        # get the wings angle using the euler angles decomposition
        # self.wings_angles_from_euler()

        # get wings angles dot
        self.wings_phi_left_dot = self.get_dot(self.wings_phi_left, sampling_rate=SAMPLING_RATE)
        self.wings_phi_right_dot = self.get_dot(self.wings_phi_right, sampling_rate=SAMPLING_RATE)

        self.wings_theta_left_dot = self.get_dot(self.wings_theta_left, sampling_rate=SAMPLING_RATE)
        self.wings_theta_right_dot = self.get_dot(self.wings_theta_right, sampling_rate=SAMPLING_RATE)

        self.wings_psi_left_dot = self.get_dot(self.wings_psi_left, sampling_rate=SAMPLING_RATE)
        self.wings_psi_right_dot = self.get_dot(self.wings_psi_right, sampling_rate=SAMPLING_RATE)

        # plt.plot(self.wings_phi_right, label="wings_phi_right")
        # plt.plot(self.wings_phi_left, label="wings_phi_left")
        # plt.plot(self.roll_angle)
        # plt.legend()
        # plt.show()

        self.omega_lab, self.omega_body, self.angular_speed_lab, self.angular_speed_body = self.get_angular_velocities(
            self.x_body, self.y_body,
            self.z_body, self.first_y_body_frame,
            self.end_frame)
        self.omega_body_dot, self.angular_acceleration_body = self.get_omega_body_dot(self.omega_body)
        try:
            self.left_amplitudes, self.right_amplitudes, self.left_half_wingbits, self.right_half_wingbits = self.get_half_wingbits_objects()
            self.left_full_wingbits, self.right_full_wingbits = self.get_full_wingbits_objects()
        except:
            self.left_amplitudes, self.right_amplitudes, self.left_half_wingbits, self.right_half_wingbits = nans_array = np.full(
                (4, 1), np.nan)
            self.left_full_wingbits, self.right_full_wingbits = nans_array = np.full((2, 1), np.nan)
        self.left_wing_peaks, self.wingbit_frequencies_left, self.wingbit_average_frequency_left = self.get_waving_frequency(
            self.wings_phi_left)
        self.left_wing_right, self.wingbit_frequencies_right, self.wingbit_average_frequency_right = self.get_waving_frequency(
            self.wings_phi_right)

        # self.omega_body_dot = self.get_omega_body_dot(self.omega_body)

        # plt.plot(self.p_dot, label='p')
        # plt.plot(self.q_dot, label='q')
        # plt.plot(self.r_dot, label='r')
        # plt.plot(self.omega_body_dot * 100)
        # plt.legend()
        # plt.grid()
        # plt.show()
        if not validation:
            if find_auto_correlation:
                self.auto_correlation_axis_angle = self.get_auto_correlation_axis_angle(self.x_body, self.y_body,
                                                                                        self.z_body,
                                                                                        self.first_y_body_frame,
                                                                                        self.end_frame)
                self.auto_correlation_x_body = self.get_auto_correlation_x_body(self.x_body)

            if create_html:
                save_path_html = os.path.join(self.dir , 'movie_html.html')
                Visualizer.create_movie_plot(com=self.center_of_mass, x_body=self.x_body, y_body=self.y_body,
                                             points_3D=self.points_3D,
                                             start_frame=self.first_y_body_frame, save_path=save_path_html)
            if create_h5:
                create_movie_analysis_h5(movie=os.path.basename(self.dir),
                                         movie_dir=self.dir,
                                         points_3D_path=self.points_3D_path,
                                         smooth=True, analysis_object=self)
            if create_mp4:
                self.create_mp4_from_movie()
        self.adjust_starting_frame()
        # self.frames_confidence_score = self.get_frames_confidence_score()
        self.omega_x, self.omega_y, self.omega_z = self.omega_body.T
        # if not validation:
        #     force_body, force_lab, torque_body =  self.get_wings_forces()
        #     self.force_body_left_wing, self.force_body_right_wing = force_body[:, LEFT, :], force_body[:, RIGHT, :]
        #     self.force_lab_left_wing, self.force_lab_right_wing = force_lab[:, LEFT, :], force_lab[:, RIGHT, :]
        #     self.torque_body_left, self.torque_body_right = torque_body[:, LEFT, :], torque_body[:, RIGHT, :]
        #     self.full_body_wing_bits = self.get_FullWingBitBody_objects()

    def calculate_deformation_angle(self):
        left_deformation_angle = FlightAnalysis.calculate_angle(self.left_wing_chord, self.left_wing_lower_chord)
        right_deformation_angle = FlightAnalysis.calculate_angle(self.right_wing_chord, self.right_wing_lower_chord)
        return left_deformation_angle, right_deformation_angle


    def get_wings_lower_cords(self):
        wings_CM = np.array([self.left_wing_CM, self.right_wing_CM])
        spans = [self.left_wing_span, self.right_wing_span]
        lower_cords_left_right = []
        for wing in range(2):
            cener_wing_point = self.points_3D[:, 6 + wing*self.num_points_per_wing, :]
            span_vector = spans[wing]
            lower_wing_point = self.points_3D[:, 4 + wing*self.num_points_per_wing, :]
            lower_chord = FlightAnalysis.shortest_vector_to_line(points_on_line=cener_wing_point,
                                                                 line_directions=span_vector,
                                                                 query_points=lower_wing_point)
            approximation = (cener_wing_point - lower_wing_point)/np.linalg.norm(cener_wing_point - lower_wing_point, axis=-1)[:, np.newaxis]
            if min(FlightAnalysis.row_wize_dot(approximation, lower_chord)) < 0:
                assert "something is wrong with the lower chord"
                lower_chord = -lower_chord
            lower_cords_left_right.append(lower_chord)
        lower_cords_left, lower_cords_right = lower_cords_left_right
        return lower_cords_left, lower_cords_right

    @staticmethod
    def shortest_vector_to_line(points_on_line, line_directions, query_points):
        # Compute the vector from the line's point to the query point
        vector_to_point = points_on_line - query_points

        # Compute the dot product of vector_to_point and line_directions for each row
        dot_product = FlightAnalysis.row_wize_dot(vector_to_point, line_directions)

        # Compute the projection vector
        projection_vector = (dot_product[:, np.newaxis] * line_directions)

        # Compute the shortest vector
        shortest_vector = vector_to_point - projection_vector

        shortest_vector = shortest_vector / np.linalg.norm(shortest_vector, axis=-1)[:, np.newaxis]

        return shortest_vector


    def get_yaw_pitch_roll_from_euler(self):
        Rs = np.stack((self.x_body, self.y_body, self.z_body), axis=-1)
        yaw = FlightAnalysis.get_yaw_from_euler(Rs)
        pitch = FlightAnalysis.get_pitch_from_euler(Rs)
        roll = FlightAnalysis.get_roll_from_euler(Rs)
        return yaw, pitch, roll[self.first_y_body_frame:self.end_frame]

    @staticmethod
    def get_yaw_from_euler(Rs):
        return np.degrees(np.unwrap(np.array([np.arctan2(r[1, 0], r[0, 0]) for r in Rs])))

    @staticmethod
    def get_pitch_from_euler(Rs):
        return -np.degrees(np.unwrap(np.array([np.arcsin(-r[2, 0]) for r in Rs]), period=np.pi))

    @staticmethod
    def get_roll_from_euler(Rs):
        roll = np.degrees(np.unwrap(np.array([np.arctan2(r[2, 1], r[2, 2]) for r in Rs])))
        # Manually correct for initial offset if needed
        if roll[0] >= 180:
            roll -= 360
        return roll


    def wings_angles_from_euler(self):
        theta = np.pi/4
        stroke_planes_cor_sys = []
        # get stoke plane coordinate system
        for frame in range(self.num_frames):
            xb, yb, zb = self.x_body[frame][np.newaxis, :], self.y_body[frame][np.newaxis, :], self.z_body[frame][np.newaxis, :]
            xb_rot, yb_rot, zb_rot = self.rodrigues_rot(xb, yb, -theta), self.rodrigues_rot(yb, yb, -theta), self.rodrigues_rot(zb, yb, -theta)
            stroke_planes_cor_sys.append(np.stack((xb_rot, yb_rot, zb_rot), axis=-1))
        stroke_planes_cor_sys = np.concatenate(stroke_planes_cor_sys, axis=0)

        # rotate the wings coordinate system to the stroke plane coordinate system
        wings_spans = [self.left_wing_span, self.right_wing_span]
        wings_chords = [self.left_wing_chord, self.right_wing_chord]
        for wing in range(2):
            all_rotated_corsys = []
            for frame in range(self.num_frames):
                span = wings_spans[wing][frame]
                chord = wings_chords[wing][frame]
                wing_z = np.cross(span, chord)
                wing_cor_sys = np.stack((span, chord, wing_z)).T

                # now rotate
                R = stroke_planes_cor_sys[frame]
                rotated_wing_cor_sys = R.T @ wing_cor_sys
                all_rotated_corsys.append(rotated_wing_cor_sys)
            all_rotated_corsys = np.array(all_rotated_corsys)
            phi = FlightAnalysis.get_yaw_from_euler(all_rotated_corsys)
            real_phi = self.wings_phi_left
            plt.plot(real_phi)
            plt.plot(phi)
            plt.show()


    def get_frames_confidence_score(self):
        nan_indices = [i for i in range(self.points_3D.shape[0]) if np.isnan(self.points_3D[i]).any()]
        no_nans = np.delete(self.points_3D, nan_indices, axis=0)
        average_std, neighbors_dist, distances_std, mean_distance, points_distances = Validation.get_wings_distances_variance(no_nans)
        deltas = points_distances - mean_distance
        mean_deltas = np.abs(deltas.mean(axis=1))
        std = np.std(mean_deltas)
        # plt.hist(mean_deltas, bins=20)
        y0 = 4 * std
        plt.axhline(y=y0, color='r', linestyle='--', label=f'y = {y0}')
        plt.plot(mean_deltas, '.')
        plt.show()
        return 0

    def create_mp4_from_movie(self):
        base_path = self.dir
        mov_num = os.path.basename(self.dir)[-2:]
        for file in os.listdir(self.dir):
            if file.startswith('movie') and file.endswith('.h5'):
                box_path = os.path.join(self.dir, file)
                break
        create_gif_one_movie(base_path=base_path, mov_num=mov_num, box_path=box_path)

    @staticmethod
    def closest_index(array, value):
        closest_index = np.argmin(np.abs(array - value))
        return closest_index

    @staticmethod
    def find_middle_frame(stroke_phi):
        start_val = stroke_phi[0]
        end_val = stroke_phi[-1]
        middle_val = (start_val + end_val) / 2
        middle_frame = FlightAnalysis.closest_index(stroke_phi, middle_val)
        return middle_frame

    @staticmethod
    def get_theta_mid_frame_diff(theta, phi):
        """
        finds the middle of the upper stroke and down stroke and caculates the theta differences in this locations
        """
        down_stroke, mid_down_stroke, mid_up_strock, up_stroke = FlightAnalysis.get_wingbit_key_frames(phi)

        theta_mid_down_strock = theta[down_stroke[mid_down_stroke]]
        theta_mid_up_stroke = theta[up_stroke[mid_up_strock]]

        theta_diff = theta_mid_up_stroke - theta_mid_down_strock
        return theta_diff

    @staticmethod
    def get_phi_frontstroke_diff(left_phi, right_phi):
        """

        """
        phi_mean = (left_phi + right_phi) / 2
        down_stroke, mid_down_stroke, mid_up_strock, up_stroke = FlightAnalysis.get_wingbit_key_frames(phi_mean)
        phi_upstroke_left = left_phi[up_stroke][-1]
        phi_upstroke_right = right_phi[up_stroke][-1]
        phi_upstroke_avarage = (phi_upstroke_left + phi_upstroke_right) / 2
        phi_frontstroke_left_minus_right = phi_upstroke_left - phi_upstroke_right
        return phi_frontstroke_left_minus_right, phi_upstroke_avarage

    @staticmethod
    def get_wingbit_key_frames(phi):
        phi_max_frame = np.argmax(phi)
        down_stroke = np.arange(0, phi_max_frame)
        mid_down_stroke = FlightAnalysis.find_middle_frame(phi[down_stroke])
        up_stroke = np.arange(phi_max_frame, len(phi))
        mid_up_strock = FlightAnalysis.find_middle_frame(phi[up_stroke])
        return down_stroke, mid_down_stroke, mid_up_strock, up_stroke

    def get_wings_forces(self):
        wings_phi = [self.wings_phi_left, -self.wings_phi_right]  # phi right is taken in minus
        wings_theta = [-self.wings_theta_left, -self.wings_theta_right] # both theta are in minus
        wings_psi = [self.wings_psi_left,180 - self.wings_psi_right]  #  right psi is 180 - psi

        wings_phi_dot = [self.wings_phi_left_dot, -self.wings_phi_right_dot]
        wings_theta_dot = [-self.wings_theta_left_dot, -self.wings_theta_right_dot]
        wings_psi_dot = [self.wings_psi_left_dot, -self.wings_psi_right_dot]

        roll_all = self.roll_angle
        pitch_all = -self.pitch_angle  # minus here
        yaw_all = self.yaw_angle
        start = np.where(~np.isnan(self.wings_theta_right_dot))[0][0]
        end = np.where(~np.isnan(self.wings_theta_right_dot))[0][-1]
        speedCalc = np.array([2.5 / 1000, 0, 0])
        hinge = np.array([0.0001, 0.0, 0.0001])
        force_body = np.full((self.wings_theta_right_dot.shape[0], 2, 3), np.nan)
        force_lab = np.full((self.wings_theta_right_dot.shape[0], 2, 3), np.nan)
        torque_body = np.full((self.wings_theta_right_dot.shape[0], 2, 3), np.nan)
        for frame in range(start, end):
            for wing in range(2):
                psi, theta, phi = wings_psi[wing][frame], wings_theta[wing][frame], wings_phi[wing][frame]
                psi_dot, theta_dot, phi_dot = wings_psi_dot[wing][frame], wings_theta_dot[wing][frame], wings_phi_dot[wing][frame]
                roll, pitch, yaw = roll_all[frame], pitch_all[frame], yaw_all[frame]
                f_body_aero, f_lab_aero, t_body, r_wing2sp, r_sp2body, r_body2lab = FlightAnalysis.exctract_forces(phi, phi_dot,
                                                                                 pitch, psi,
                                                                                 psi_dot,
                                                                                 roll, theta,
                                                                                 theta_dot, yaw,
                                                                                 self.CM_dot[frame],
                                                                                 self.omega_body[frame])
                force_body_hat = f_body_aero / np.linalg.norm(f_body_aero)
                force_size = np.linalg.norm(f_lab_aero)
                force_body[frame, wing, :] = f_body_aero
                force_lab[frame, wing, :] = f_lab_aero
                torque_body[frame, wing, :] = t_body
                if wing == LEFT:
                    Rot_mat_wing2labL = r_body2lab @ r_sp2body @ r_wing2sp
                    f_body_left = f_lab_aero
                else:
                    Rot_mat_wing2labR = r_body2lab @ r_sp2body @ r_wing2sp
                    f_body_right = f_lab_aero
            # dispaly_coordinate_systems(Rot_mat_wing2labL, Rot_mat_wing2labR, r_body2lab, f_body_left, f_body_right)
        return force_body, force_lab, torque_body

    @staticmethod
    def exctract_forces(phi, phi_dot, pitch, psi, psi_dot, roll, theta, theta_dot, yaw, center_mass_dot, omega_body):
        speedCalc = np.array([2.5 / 1000, 0, 0])
        hinge = np.array([0.0001, 0.0, 0.0001])
        r_wing2sp = R.from_euler('xyz', [psi, theta, phi], degrees=True).as_matrix()  # todo wing to stroke plane
        r_sp2body = R.from_euler('xyz', [0, 45, 0], degrees=True).as_matrix()  # stroke plane to body
        r_body2lab = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        r_sp2lab = r_body2lab @ r_sp2body  # body2lab @ stroke_plane_to_body -> stroke plane to lab
        angles = np.radians([psi, theta, phi])
        angles_dot = np.radians([psi_dot, theta_dot, phi_dot])
        angular_wing_vel = FlightAnalysis.body_ang_vel_pqr(angles, angles_dot, get_pqr=True)
        tang_wing_v = np.cross(angular_wing_vel, speedCalc)

        ## calculate the body velocity in the wing coordinate system ##
        ac_lab = r_sp2lab @ r_wing2sp @ speedCalc
        ac_bod = r_body2lab.T @ ac_lab
        vel_loc_bod = ac_bod + hinge
        vb = np.cross(np.radians(omega_body), vel_loc_bod) + center_mass_dot
        vb = r_body2lab @ vb
        vb_lab = r_sp2lab.T @ vb
        vb_wing = r_wing2sp.T @ vb_lab

        vw =  vb_wing + tang_wing_v
        alpha = np.arctan2(vw[2], vw[1])  # angle of attack
        cl, cd, span_hat, lhat, drag, lift, t_body, f_body_aero, f_lab_aero, t_lab = FlightAnalysis.get_forces(
            alpha, vw,
            r_body2lab,
            r_wing2sp,
            r_sp2lab)
        return f_body_aero, f_lab_aero, t_body, r_wing2sp, r_sp2body, r_body2lab


    @staticmethod
    def body_ang_vel_pqr(angles, angles_dot, get_pqr):
        """
        Converts change in euler angles to body rates (if get_pqr is True) or body rates to euler rates (if get_pqr is False)
        :param angles: euler angles (np.array[psi,theta,phi])
        :param angles_dot: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
        :param get_pqr: whether to get body rates from euler rates or the other way around
        :return: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
        """
        psi = angles[0]
        theta = angles[1]
        psi_p_dot = angles_dot[0]
        theta_q_dot = angles_dot[1]
        phi_r_dot = angles_dot[2]

        if get_pqr:
            p = psi_p_dot - phi_r_dot * np.sin(theta)
            q = theta_q_dot * np.cos(psi) + phi_r_dot * np.sin(psi) * np.cos(theta)
            r = -theta_q_dot * np.sin(psi) + phi_r_dot * np.cos(psi) * np.cos(theta)
            return np.array([p, q, r])
        else:
            psi_dot = psi_p_dot + theta_q_dot * (np.sin(psi) * np.sin(theta)) / np.cos(theta) + phi_r_dot * (
                    np.cos(psi) * np.sin(theta)) / np.cos(theta)
            theta_dot = theta_q_dot * np.cos(psi) + phi_r_dot * -np.sin(psi)
            phi_dot = theta_q_dot * np.sin(psi) / np.cos(theta) + phi_r_dot * np.cos(psi) / np.cos(theta)
            return np.array([psi_dot, theta_dot, phi_dot])

    @staticmethod
    def get_forces(aoa, v_wing, rotation_mat_body2lab, rotation_mat_wing2sp, rotation_mat_sp2lab):
        clmax = 1.8
        cdmax = 3.4
        cd0 = 0.4
        span_hat = np.array([1, 0, 0])
        rho = 1.225
        span = 2.5 / 1000
        chord = 0.7 / 1000
        s = span * chord * np.pi / 4
        r22 = 0.4
        ac_loc = np.array([2.5/1000*0.7, 0, 0])
        hinge_location = np.array([0.0001,0.0,0.0001])

        cl = clmax * np.sin(2 * aoa)
        cd = (cdmax + cd0) / 2 - (cdmax - cd0) / 2 * np.cos(2 * aoa)
        u = v_wing[0] ** 2 + v_wing[1] ** 2 + v_wing[2] ** 2
        uhat = v_wing / np.linalg.norm(v_wing)
        lhat = (np.cross(span_hat, -uhat)).T  # perpendicular to Uhat
        lhat = lhat / np.linalg.norm(lhat)
        q = rho * s * r22 * u
        drag = -0.5 * cd * q * uhat
        lift = 0.5 * cl * q * lhat
        rot_mat_spw2lab = rotation_mat_sp2lab @ rotation_mat_wing2sp
        ac_loc_lab = rot_mat_spw2lab @ ac_loc.T + rotation_mat_body2lab @ hinge_location.T  # AC location in lab axes
        ac_loc_body = rotation_mat_body2lab.T @ ac_loc_lab  # AC location in body axes

        f_lab_aero = rot_mat_spw2lab @ lift + rot_mat_spw2lab @ drag
        # force in body axes
        f_body = rotation_mat_body2lab.T @ f_lab_aero
        t_lab = np.cross(ac_loc_lab.T,
                         f_lab_aero).T  # + cross(ACLocB_body, Dbod).T # torque on body (in body axes)
        # (from forces, no CM0)
        t_body = np.cross(ac_loc_body.T,
                          f_body).T  # + cross(ACLocB_lab, Dbod_lab) # torque on body( in bodyaxes)
        # (from forces, no CM0)
        return cl, cd, span_hat, lhat, drag, lift, t_body, f_body, f_lab_aero, t_lab

    @staticmethod
    def get_psi_attributes(psi, phi):
        down_stroke, mid_down_stroke, mid_up_strock, up_stroke = FlightAnalysis.get_wingbit_key_frames(phi)
        psi_mid_downstrock = psi[down_stroke[mid_down_stroke]]
        psi_mid_upstroke = psi[up_stroke[mid_up_strock]]
        psi_diff = psi_mid_upstroke - psi_mid_downstrock
        return psi_mid_downstrock, psi_mid_upstroke, psi_diff

    def get_FullWingBitBody_objects(self):
        left_phi = self.wings_phi_left
        right_phi = self.wings_phi_right
        avg_phi = (left_phi + right_phi) / 2

        # plt.plot(left_phi, label="left_phi")
        # plt.plot(right_phi, label="right_phi")
        # plt.plot(avg_phi, label="avg_phi")
        # plt.legend()
        # plt.show()

        _, max_peaks_frames_left, _, min_peaks_frames_left = self.get_phi_peaks(left_phi)
        _, max_peaks_frames_right, _, min_peaks_frames_right = self.get_phi_peaks(right_phi)

        max_peak_values, max_peaks_inds, min_peak_values, min_peaks_inds = self.get_phi_peaks(avg_phi)

        min_peaks_frames = np.round(min_peaks_inds).astype(int)
        max_peaks_frames_left = np.round(max_peaks_frames_left).astype(int)
        min_peaks_frames_left = np.round(min_peaks_frames_left).astype(int)
        max_peaks_frames_right = np.round(max_peaks_frames_right).astype(int)
        min_peaks_frames_right = np.round(min_peaks_frames_right).astype(int)

        full_wingbits_objects = []

        for i in range(len(min_peaks_frames) - 1):
            # print(i)
            start_frame = min_peaks_frames[i]
            end_frame = min_peaks_frames[i + 1]
            frames = np.arange(start_frame, end_frame + 1)

            # Calculate left and right amplitudes
            max_frame_left = \
                max_peaks_frames_left[(max_peaks_frames_left > start_frame) & (max_peaks_frames_left < end_frame)][0]
            max_frame_right = \
                max_peaks_frames_right[(max_peaks_frames_right > start_frame) & (max_peaks_frames_right < end_frame)][0]

            # phi
            left_phi_max = left_phi[max_frame_left]
            left_phi_min = left_phi[start_frame]
            right_phi_max = right_phi[max_frame_right]
            right_phi_min = right_phi[start_frame]
            phi_max_average = (left_phi_max + right_phi_max) / 2
            phi_min_average = (left_phi_min + right_phi_min) / 2
            left_amplitude = left_phi[max_frame_left] - left_phi[start_frame]
            right_amplitude = right_phi[max_frame_right] - right_phi[start_frame]
            left_phi_middle = (left_phi_max + left_phi_min) / 2
            right_phi_middle = (right_phi_max + right_phi_min) / 2
            phi_middle_average = (left_phi_middle + right_phi_middle) / 2
            phi_frontstroke_left_minus_right, phi_upstroke_avarage = FlightAnalysis.get_phi_frontstroke_diff(
                left_phi[frames], right_phi[frames])

            # theta
            left_theta_mid_frame_dif = FlightAnalysis.get_theta_mid_frame_diff(theta=self.wings_theta_left[frames],
                                                                               phi=self.wings_phi_left[frames])
            right_theta_mid_frame_dif = FlightAnalysis.get_theta_mid_frame_diff(theta=self.wings_theta_right[frames],
                                                                                phi=self.wings_phi_right[frames])
            avarage_theta_mid_frame_dif = (left_theta_mid_frame_dif + right_theta_mid_frame_dif) / 2

            # psi for every wing mid upstroke, mid downsroke, difference,
            left_psi = self.wings_psi_left[frames]
            right_psi = self.wings_psi_right[frames]
            left_psi_mid_downstroke, left_psi_mid_upstroke, left_psi_diff = FlightAnalysis.get_psi_attributes(
                psi=self.wings_psi_left[frames],
                phi=self.wings_phi_left[frames])
            right_psi_mid_downstroke, right_psi_mid_upstroke, right_psi_diff = FlightAnalysis.get_psi_attributes(
                psi=self.wings_psi_right[frames],
                phi=self.wings_phi_right[frames])
            left_minus_right_psi_mid_downstroke = left_psi_mid_downstroke - right_psi_mid_downstroke
            left_minus_right_psi_mid_upstroke = left_psi_mid_upstroke - right_psi_mid_upstroke

            average_psi_mid_downstroke = (left_psi_mid_downstroke + right_psi_mid_downstroke) / 2
            average_psi_mid_upstroke = (right_psi_mid_downstroke + right_psi_mid_upstroke) / 2
            average_psi_diff = (left_psi_diff + right_psi_diff) / 2

            # torque
            torque_body_total = self.torque_body_right + self.torque_body_left
            avarage_torque_body = np.mean(torque_body_total[frames], axis=0)

            full_wingbit_body = FullWingBitBody(
                start=np.round(start_frame).astype(int),
                end=np.round(end_frame).astype(int),
                frames=frames,
                points_3D=self.points_3D[frames],
                CM_dot=self.CM_dot[frames],
                CM_speed=self.CM_speed[frames],
                yaw_angle=self.yaw_angle[frames],
                pitch_angle=self.pitch_angle[frames],
                roll_angle=self.roll_angle[frames],
                roll_dot=self.roll_dot[frames],
                pitch_dot=self.pitch_dot[frames],
                yaw_dot=self.yaw_dot[frames],
                omega_body=self.omega_body[frames],
                angular_speed_body=self.angular_speed_body[frames],
                p=self.p[frames],
                q=self.q[frames],
                r=self.r[frames],
                torque=torque_body_total[frames],

                avarage_torque_body=avarage_torque_body,

                # phi attributes
                left_amplitude=left_amplitude,
                right_amplitude=right_amplitude,
                left_phi_max=left_phi_max,
                left_phi_min=left_phi_min,
                right_phi_max=right_phi_max,
                right_phi_min=right_phi_min,
                phi_max_average=phi_max_average,
                phi_min_average=phi_min_average,
                phi_mid_frame_left=left_phi_middle,
                phi_mid_frame_right=right_phi_middle,
                phi_mid_frame_average=phi_middle_average,
                phi_frontstroke_left_minus_right=phi_frontstroke_left_minus_right,
                phi_upstroke_avarage=phi_upstroke_avarage,

                # theta attributrs
                left_theta_mid_frame_dif=left_theta_mid_frame_dif,
                right_theta_mid_frame_dif=right_theta_mid_frame_dif,
                avarage_theta_mid_frame_dif=avarage_theta_mid_frame_dif,

                # psi attribues
                left_psi_mid_downstroke=left_psi_mid_downstroke,
                left_psi_mid_upstroke=left_psi_mid_upstroke,
                left_psi_diff=left_psi_diff,
                right_psi_mid_downstroke=right_psi_mid_downstroke,
                right_psi_mid_upstroke=right_psi_mid_upstroke,
                right_psi_diff=right_psi_diff,
                average_psi_mid_downstroke=average_psi_mid_downstroke,
                average_psi_mid_upstroke=average_psi_mid_upstroke,
                average_psi_diff=average_psi_diff,
                left_minus_right_psi_mid_downstroke=left_minus_right_psi_mid_downstroke,
                left_minus_right_psi_mid_upstroke=left_minus_right_psi_mid_upstroke,

                roll_dot_dot=self.roll_dot_dot[frames],
                yaw_dot_dot=self.yaw_dot_dot[frames],
                pitch_dot_dot=self.pitch_dot_dot[frames],
                omega_body_dot=self.omega_body_dot[frames],
                angular_acceleration_body=self.angular_acceleration_body[frames],

                # wings dot attributes
                wings_phi_left_dot=self.wings_phi_left_dot[frames],
                wings_phi_right_dot=self.wings_phi_right_dot[frames],

                wings_theta_left_dot=self.wings_theta_left_dot[frames],
                wings_theta_right_dot=self.wings_theta_right_dot[frames],

                wings_psi_left_dot=self.wings_psi_left_dot[frames],
                wings_psi_right_dot=self.wings_psi_right_dot[frames]
            )
            full_wingbits_objects.append(full_wingbit_body)

        return full_wingbits_objects

    def get_omega_body_dot(self, omega_body):
        omega_body_dot = np.full(omega_body.shape, np.nan)
        N, axis = omega_body.shape
        for ax in range(axis):
            ax_dot = self.get_dot(data=omega_body[self.first_y_body_frame:self.end_frame, ax],
                                  sampling_rate=SAMPLING_RATE)
            omega_body_dot[self.first_y_body_frame:self.end_frame, ax] = ax_dot
        angular_acceleration_body = np.linalg.norm(omega_body_dot, axis=-1)
        return omega_body_dot, angular_acceleration_body

    def get_pqr(self):
        # calculate the body angular acceleratrion and velocity: pqr and pqr_dot
        roll, roll_dot, roll_dot_dot = np.zeros(self.num_frames), np.zeros(self.num_frames), np.zeros(self.num_frames)
        pitch = -self.pitch_angle  # there is minus here
        yaw = self.yaw_angle
        roll[self.first_y_body_frame:self.end_frame] = self.roll_angle
        pitch_dot = -self.pitch_dot  # there is minus here
        yaw_dot = self.yaw_dot
        roll_dot[self.first_y_body_frame:self.end_frame] = self.roll_dot
        pitch_dot_dot = self.pitch_dot_dot
        yaw_dot_dot = self.yaw_dot_dot
        roll_dot_dot[self.first_y_body_frame:self.end_frame] = self.roll_dot_dot

        p, q, r = self.get_pqr_calculation(pitch, pitch_dot, roll, roll_dot, yaw_dot)
        p, q, r = np.degrees(p), np.degrees(q), np.degrees(r)

        # p,q,r dot

        # p_dot = roll_dot_dot - yaw_dot_dot * np.sin(np.radians(pitch)) - yaw_dot * pitch_dot * np.cos(
        #     np.radians(pitch));
        # q_dot = (pitch_dot_dot * np.cos(np.radians(roll)) - pitch_dot * np.sin(np.radians(roll)) * roll_dot
        #          + yaw_dot_dot * np.sin(np.radians(roll) * np.cos(pitch) +
        #                                 +yaw_dot * np.cos(np.radians(roll)) * np.cos(np.radians(pitch)) * roll_dot
        #                                 - yaw_dot * np.sin(np.radians(roll)) * np.sin(np.radians(pitch)) * pitch_dot))
        #
        # r_dot = (-pitch_dot_dot * np.sin(np.radians(roll)) - pitch_dot * np.cos(np.radians(roll)) * roll_dot
        #          + yaw_dot_dot * np.cos(np.radians(roll)) * np.cos(np.radians(pitch))
        #          - yaw_dot * np.sin(np.radians(roll)) * np.cos(np.radians(pitch)) * roll_dot
        #          - yaw_dot * np.cos(np.radians(roll)) * np.sin(np.radians(pitch)) * pitch_dot)

        return p, q, r,  # p_dot, q_dot, r_dot

    @staticmethod
    def get_pqr_calculation(pitch, pitch_dot, roll, roll_dot, yaw_dot):
        # get everything in degrees.
        # return in radians
        pitch, pitch_dot, roll, roll_dot, yaw_dot = (np.radians(pitch), np.radians(pitch_dot),
                                                     np.radians(roll), np.radians(roll_dot),
                                                     np.radians(yaw_dot))
        p = roll_dot - yaw_dot * np.sin(pitch)
        q = pitch_dot * np.cos(roll) + yaw_dot * np.sin(roll) * np.cos(pitch)
        r = -pitch_dot * np.sin(roll) + yaw_dot * np.cos(roll) * np.cos(pitch)
        return p, q, r

    def get_roll_dot(self, roll_angle):
        roll_dot = self.get_dot(roll_angle, sampling_rate=SAMPLING_RATE)
        return roll_dot

    def get_full_wingbits_objects(self):
        all_half_wingbits_objects = [self.left_half_wingbits, self.right_half_wingbits]
        phi_wings = [self.wings_phi_left, self.wings_phi_right]
        theta_wings = [self.wings_theta_left, self.wings_theta_right]
        psi_wings = [self.wings_psi_left, self.wings_psi_right]
        full_wingbits_objects = [[], []]
        for wing in range(2):
            half_wingbits = all_half_wingbits_objects[wing]
            for i in range(0, len(half_wingbits), 2):
                first_half = half_wingbits[i]
                if i + 1 < len(half_wingbits):
                    second_half = half_wingbits[i + 1]
                    start = first_half.start
                    end = second_half.end
                    frames = np.unique(np.concatenate((first_half.frames, second_half.frames)))
                    wingbit_phi = phi_wings[wing][frames - self.first_analysed_frame]
                    wingbit_theta = theta_wings[wing][frames - self.first_analysed_frame]
                    wingbit_psi = psi_wings[wing][frames - self.first_analysed_frame]
                    full_wingbit = FullWingBit(start=start,
                                               end=end,
                                               frames=frames,
                                               phi_vals=wingbit_phi,
                                               theta_vals=wingbit_theta,
                                               psi_vals=wingbit_psi)
                    full_wingbits_objects[wing].append(full_wingbit)
                else:
                    # Handle the case where there is no second half
                    start = first_half.start
                    end = first_half.end
                    frames = first_half.frames
                    wingbit_phi = phi_wings[wing][frames - self.first_analysed_frame]
                    wingbit_theta = theta_wings[wing][frames - self.first_analysed_frame]
                    wingbit_psi = psi_wings[wing][frames - self.first_analysed_frame]
                    full_wingbit = FullWingBit(start=start,
                                               end=end,
                                               frames=frames,
                                               phi_vals=wingbit_phi,
                                               theta_vals=wingbit_theta,
                                               psi_vals=wingbit_psi)
                    full_wingbits_objects[wing].append(full_wingbit)

        left_full_wingbits, right_full_wingbits = full_wingbits_objects
        return left_full_wingbits, right_full_wingbits

    def get_half_wingbits_objects(self):
        phi_wings = [self.wings_phi_left, self.wings_phi_right]
        half_wingbits = [[], []]
        phi_amplitudes = [[], []]
        for wing in range(2):
            phi = phi_wings[wing]
            max_peak_values, max_peaks_frames, min_peak_values, min_peaks_frames = self.get_phi_peaks(phi)
            max_peaks_frames = np.array(max_peaks_frames) + self.first_analysed_frame
            min_peaks_frames = np.array(min_peaks_frames) + self.first_analysed_frame
            phi = FlightAnalysis.add_nan_frames(phi, self.first_analysed_frame)
            max_peaks = np.column_stack((max_peaks_frames, max_peak_values))
            min_peaks = np.column_stack((min_peaks_frames, min_peak_values))
            all_peaks = np.concatenate((max_peaks, min_peaks), axis=0)
            all_peaks = all_peaks[all_peaks[:, 0].argsort()]
            all_amplitudes = np.abs(np.diff(all_peaks[:, 1]))
            if np.min(all_amplitudes) < 100:
                assert f"need to check minimum amplitude in phi wing {wing}"
            for cur_peak_ind in range(len(all_peaks) - 1):
                next_peak_ind = cur_peak_ind + 1
                cur_peak_val = all_peaks[cur_peak_ind, 1]
                next_peak_val = all_peaks[next_peak_ind, 1]
                average_val = (cur_peak_val + next_peak_val) / 2
                half_bit_start = np.round(all_peaks[cur_peak_ind, 0]).astype(int)
                half_bit_end = np.round(all_peaks[next_peak_ind, 0]).astype(int)
                halfbit_frames = np.arange(half_bit_start, half_bit_end)
                strock_direction = "back_strock" if next_peak_val > cur_peak_val else "up_strock"
                amplitude = np.abs(cur_peak_val - next_peak_val)
                mid_index = (all_peaks[cur_peak_ind, 0] + all_peaks[next_peak_ind, 0]) / 2
                try:
                    half_wingbit_obj = HalfWingbit(start=all_peaks[cur_peak_ind, 0],
                                                   end=all_peaks[next_peak_ind, 0],
                                                   frames=halfbit_frames,
                                                   phi_vals=phi[halfbit_frames],
                                                   start_peak_val=cur_peak_val,
                                                   end_peak_val=next_peak_val,
                                                   average_value=average_val,
                                                   strock_direction=strock_direction)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                half_wingbits[wing].append(half_wingbit_obj)
                phi_amplitudes[wing].append((mid_index, amplitude))
        left_half_wingbits, right_half_wingbits = half_wingbits
        left_wing_amplitudes, right_wing_amplitudes = np.array(phi_amplitudes[0]), np.array(phi_amplitudes[1])
        return left_wing_amplitudes, right_wing_amplitudes, left_half_wingbits, right_half_wingbits

    @staticmethod
    def add_nan_frames(original_array, N):
        nan_frames = np.full((N,) + original_array.shape[1:], np.nan)
        new_array = np.concatenate((nan_frames, original_array), axis=0)
        return new_array

    @staticmethod
    def fill_with_nans(original_array, indices):
        nan_frames = np.full((len(indices),) + original_array.shape[1:], np.nan)
        original_array[indices, ...] = nan_frames
        return original_array

    def adjust_starting_frame(self):
        # fill all the not analysed frames with nans
        # fill with nans all arrays that rely on y_body
        indices = np.concatenate(
            (np.arange(0, self.first_y_body_frame), np.arange(self.end_frame, self.num_frames - 1)))

        # fix roll angle and roll dot
        roll_angle, roll_dot, roll_dot_dot = np.zeros((3, self.num_frames))
        roll_angle[self.first_y_body_frame:self.end_frame] = self.roll_angle
        roll_dot[self.first_y_body_frame:self.end_frame] = self.roll_dot
        roll_dot_dot[self.first_y_body_frame:self.end_frame] = self.roll_dot_dot
        self.roll_angle = roll_angle
        self.roll_dot = roll_dot
        self.roll_dot_dot = roll_dot_dot

        self.p = FlightAnalysis.fill_with_nans(self.p, indices)
        self.q = FlightAnalysis.fill_with_nans(self.q, indices)
        self.r = FlightAnalysis.fill_with_nans(self.r, indices)
        self.roll_angle = FlightAnalysis.fill_with_nans(self.roll_angle, indices)
        self.roll_dot = FlightAnalysis.fill_with_nans(self.roll_dot, indices)
        self.roll_dot_dot = FlightAnalysis.fill_with_nans(self.roll_dot_dot, indices)
        self.y_body = FlightAnalysis.fill_with_nans(self.y_body, indices)
        self.z_body = FlightAnalysis.fill_with_nans(self.z_body, indices)
        self.omega_lab = FlightAnalysis.fill_with_nans(self.omega_lab, indices)
        self.omega_body = FlightAnalysis.fill_with_nans(self.omega_body, indices)
        self.omega_body_dot = FlightAnalysis.fill_with_nans(self.omega_body_dot, indices)
        self.angular_acceleration_body = FlightAnalysis.fill_with_nans(self.angular_acceleration_body, indices)
        self.angular_speed_body = FlightAnalysis.fill_with_nans(self.angular_speed_body, indices)
        self.stroke_planes = FlightAnalysis.fill_with_nans(self.stroke_planes, indices)
        self.wings_phi_left = FlightAnalysis.fill_with_nans(self.wings_phi_left, indices)
        self.wings_phi_right = FlightAnalysis.fill_with_nans(self.wings_phi_right, indices)
        self.wings_theta_right = FlightAnalysis.fill_with_nans(self.wings_theta_right, indices)
        self.wings_theta_left = FlightAnalysis.fill_with_nans(self.wings_theta_left, indices)
        self.wings_psi_right = FlightAnalysis.fill_with_nans(self.wings_psi_right, indices)
        self.wings_psi_left = FlightAnalysis.fill_with_nans(self.wings_psi_left, indices)

        self.wings_phi_left_dot = FlightAnalysis.fill_with_nans(self.wings_phi_left_dot, indices)
        self.wings_phi_right_dot = FlightAnalysis.fill_with_nans(self.wings_phi_right_dot, indices)
        self.wings_theta_left_dot = FlightAnalysis.fill_with_nans(self.wings_theta_left_dot, indices)
        self.wings_theta_right_dot = FlightAnalysis.fill_with_nans(self.wings_theta_right_dot, indices)
        self.wings_psi_left_dot = FlightAnalysis.fill_with_nans(self.wings_psi_left_dot, indices)
        self.wings_psi_right_dot = FlightAnalysis.fill_with_nans(self.wings_psi_right_dot, indices)

        # add nan frames before the starting frame of the analysis
        attributes = [
            "points_3D", "head_tail_points", "x_body", "y_body", "z_body",
            "wings_tips_left", "wings_tips_right", "left_wing_CM", "right_wing_CM", "wings_joints_points",
            "CM_dot", "CM_speed",
            "left_wing_span", "right_wing_span", "left_wing_chord", "right_wing_chord",
            "all_2_planes", "all_upper_planes", "all_lower_planes", "wings_span_vecs",
            "wings_joints_vec", "wings_joints_vec_smoothed", "yaw_angle", "pitch_angle", "roll_angle",
            "roll_dot", "pitch_dot", "yaw_dot", "roll_dot_dot", "pitch_dot_dot", "yaw_dot_dot"
            , "stroke_planes", "center_of_mass", "body_speed",
            "wing_tips_speed", "wings_phi_left", "wings_phi_right", "wings_theta_left", "wings_theta_right",
            "wings_psi_left", "wings_psi_right",
            "omega_lab", "omega_body", "omega_body_dot", "angular_speed_lab", "angular_speed_body",
            "angular_acceleration_body", "p", "q", "r",
            "wings_phi_left_dot", "wings_phi_right_dot", "wings_theta_left_dot",
            "wings_theta_right_dot", "wings_psi_left_dot", "wings_psi_right_dot", "wings_chord",
            "left_wing_lower_chord", "right_wing_lower_chord",
            "left_deformation_angle", "right_deformation_angle",
        ]

        for attr in attributes:
            if hasattr(self, attr):
                if attr not in ["left_half_wingbits", "self.right_half_wingbits",
                                "left_amplitudes", "right_amplitudes"]:
                    data = getattr(self, attr)
                    # Call the add_nan_frames method and update the attribute
                    updated_data = FlightAnalysis.add_nan_frames(data, self.first_analysed_frame)
                    setattr(self, attr, updated_data)

    def find_cut_value(self):
        cut_pattern = re.compile(r'cut:\s*(\d+)')
        movie_dir_path = os.path.dirname(self.points_3D_path)
        for filename in os.listdir(movie_dir_path):
            if filename.startswith("README_mov"):
                readme_file = os.path.join(movie_dir_path, filename)
                with open(readme_file, 'r') as file:
                    for line in file:
                        match = cut_pattern.search(line)
                        if match:
                            return int(match.group(1))
        return None

    def get_first_analysed_frame(self):
        directory_path = os.path.dirname(os.path.realpath(self.points_3D_path))
        start_frame = get_start_frame(directory_path)
        return int(start_frame)

    def find_if_flip(self):
        directory_path = os.path.dirname(os.path.realpath(self.points_3D_path))
        flip = find_flip_in_files(directory_path)
        return flip

    @staticmethod
    def low_pass_filter(data, cutoff, fs):
        """
        Apply a low-pass filter to the data array.

        Parameters:
            data (np.array): Input data array.
            cutoff (float): Cutoff frequency in Hz.
            fs (int): Sampling frequency of the data.

        Returns:
            np.array: Filtered data.
        """
        n = len(data)  # length of the data
        freq = np.fft.fftfreq(n, d=1 / fs)  # frequency array

        # Fourier transform of the data
        data_fft = np.fft.fft(data)

        # Create a mask for frequencies higher than the cutoff
        mask = abs(freq) > cutoff

        # Apply mask to the Fourier spectrum
        data_fft[mask] = 0

        # Inverse Fourier Transform to get back to time domain
        filtered_data = np.fft.ifft(data_fft)

        return filtered_data.real  # Return the real part of the complex output

    @staticmethod
    def savgol_smoothing(input_points, lam, polyorder, window_length, median_kernel=1, plot=False):
        points_smoothed = np.zeros_like(input_points)
        for pnt in range(input_points.shape[1]):
            points_orig = input_points[:, pnt, :]
            points = np.apply_along_axis(medfilt, axis=0, arr=points_orig, kernel_size=median_kernel)
            A = np.arange(len(points))
            for axis in range(3):
                points[:, axis] = savgol_filter(np.copy(points[:, axis]), window_length, polyorder)
                spline = make_smoothing_spline(A, points[:, axis], lam=lam * len(A))
                points[:, axis] = spline(A)
            if plot:
                plt.plot(points_orig)
                plt.plot(points)
                plt.show()
            points_smoothed[:, pnt, :] = points
        return points_smoothed

    def get_estimated_chords(self):
        cord_estimates = []
        for wing in range(2):
            leading_edge_point = (self.points_3D[:, 0 + wing * self.num_points_per_wing, :] +
                                  self.points_3D[:, 1 + wing * self.num_points_per_wing, :]) / 2
            center_of_wing_point = (self.points_3D[:, 6 + wing * self.num_points_per_wing, :] +
                                    self.points_3D[:, 2 + wing * self.num_points_per_wing, :]) / 2
            chord_estimate = leading_edge_point - center_of_wing_point
            chord_estimate /= np.linalg.norm(chord_estimate, axis=1)[:, np.newaxis]
            cord_estimates.append(chord_estimate)
        chord_estimate_left, chord_estimate_right = cord_estimates
        return chord_estimate_left, chord_estimate_right

    def get_wings_cords(self):
        wings_spans = np.concatenate((self.left_wing_span[:, np.newaxis, :],
                                      self.right_wing_span[:, np.newaxis, :]), axis=1)
        wing_reference_points = [[4, 1], [12, 9]]
        estimated_chords_left, estimated_chords_right = self.get_estimated_chords()
        estimated_chords = [estimated_chords_left, estimated_chords_right]
        wings_chords = np.zeros_like(wings_spans)
        for frame in range(self.num_frames):
            for wing in range(2):
                span = wings_spans[frame, wing, :]
                wing_plane_normal = self.all_upper_planes[frame, wing, :-1]
                chord = np.cross(span, wing_plane_normal)
                chord /= np.linalg.norm(chord)
                # check direction
                estimated_chord = estimated_chords[wing][frame]
                if np.dot(chord, estimated_chord) < 0:
                    chord = -chord
                # make sure chord is
                wings_chords[frame, wing, :] = chord
        return wings_chords[:, LEFT], wings_chords[:, RIGHT]

    def get_wings_psi(self):
        all_psi = np.zeros((2, self.num_frames))
        all_psi_rad = np.zeros((2, self.num_frames))  # Store radians here for unwrapping

        for frame in range(self.first_y_body_frame, self.num_frames):
            for wing in range(2):
                strkpln = self.stroke_planes[frame, :-1]
                if wing == RIGHT:
                    spn = self.right_wing_span[frame]
                    chord = self.right_wing_chord[frame]
                    signy = 1
                else:
                    spn = self.left_wing_span[frame]
                    chord = self.left_wing_chord[frame]
                    signy = -1

                Surf_sp = np.cross(strkpln, spn)
                Surf_sp /= np.linalg.norm(Surf_sp)
                yax = np.cross(spn, Surf_sp)
                yax /= np.linalg.norm(yax)
                ypsi = signy * np.dot(chord, yax)
                xpsi = np.dot(chord, Surf_sp)
                angtmp = np.arctan2(ypsi, xpsi)
                all_psi_rad[wing, frame] = angtmp  # Store radians

        # Unwrap radians to prevent jumps
        all_psi_rad = np.unwrap(all_psi_rad, axis=1)

        # Convert radians to degrees
        all_psi_deg = np.rad2deg(all_psi_rad)

        # Split into left and right wing psi
        left_wing_psi, right_wing_psi = all_psi_deg

        left_wing_psi = 180 + left_wing_psi

        left_wing_psi = medfilt(left_wing_psi)
        right_wing_psi = medfilt(right_wing_psi)

        # plt.plot(left_wing_psi)
        # plt.plot(right_wing_psi)
        # plt.show()
        return left_wing_psi, right_wing_psi

    def get_head_tail_points(self, smooth=True):
        head_tail_points = self.points_3D[:, self.head_tail_inds, :]
        if smooth:
            window_length = min(73 * 3, self.num_frames)
            window_length = window_length - 1 if window_length % 2 == 0 else window_length
            median_kernel = min(11, self.num_frames)
            median_kernel = median_kernel - 1 if median_kernel % 2 == 0 else median_kernel
            head_tail_smoothed = FlightAnalysis.savgol_smoothing(head_tail_points, lam=100, polyorder=2,
                                                                 window_length=window_length,
                                                                 median_kernel=median_kernel)

            # plt.plot(np.mean(head_tail_points, axis=1))
            # plt.plot(np.mean(head_tail_smoothed, axis=1))
            # plt.show()
            head_tail_points = head_tail_smoothed
        return head_tail_points

    def get_wings_joints(self, smooth=True):
        wings_joints = self.points_3D[:, WINGS_JOINTS_INDS, :]
        if smooth:
            window_length = min(73 * 3, self.num_frames)
            window_length = window_length - 1 if window_length % 2 == 0 else window_length
            median_kernel = min(41, self.num_frames)
            median_kernel = median_kernel - 1 if median_kernel % 2 == 0 else median_kernel
            wings_joints_smoothed = FlightAnalysis.savgol_smoothing(wings_joints, lam=100, polyorder=1,
                                                                    window_length=window_length,
                                                                    median_kernel=median_kernel)
            wings_joints = wings_joints_smoothed
        return wings_joints

    def get_wings_CM(self):
        CM_left = np.mean(self.points_3D[:, self.left_wing_inds, :], axis=1)
        CM_right = np.mean(self.points_3D[:, self.right_wing_inds, :], axis=1)
        return CM_left, CM_right

    def get_wing_tips(self):
        wing_tips = self.points_3D[:, self.wings_tips_inds, :]
        wings_tips_left, wings_tips_right = wing_tips[:, LEFT, :], wing_tips[:, RIGHT, :]
        return wings_tips_left, wings_tips_right

    def get_center_of_mass(self):
        CM = np.mean(self.head_tail_points, axis=1)
        return CM

    def get_body_speed(self):
        return self.get_speed(self.center_of_mass)

    def get_wing_tips_speed(self):
        left_tip_speed, _ = self.get_speed(self.wings_tips_left[:, :])
        right_tip_speed, _ = self.get_speed(self.wings_tips_right[:, :])
        wing_tips_speed = np.concatenate((right_tip_speed[:, np.newaxis], left_tip_speed[:, np.newaxis]), axis=1)
        return wing_tips_speed

    @staticmethod
    def get_speed(points_3d, sampling_rate=SAMPLING_RATE):
        T = np.arange(len(points_3d))
        derivative_3D = np.zeros((len(points_3d), 3))
        for axis in range(3):
            derivative_3D[:, axis] = FlightAnalysis.get_dot(points_3d[:, axis], sampling_rate=sampling_rate)
        speed = np.linalg.norm(derivative_3D, axis=1)
        return speed, derivative_3D

    def get_wings_spans(self):
        # left_wing_span, right_wing_span = self.get_spans_centers_to_tip()
        left_wing_span, right_wing_span = self.get_spans_leading_edge()
        return left_wing_span, right_wing_span

    def get_spans_leading_edge(self):
        wing_spans = np.zeros((self.num_frames, 2, 3))
        wings_CM = np.array([self.left_wing_CM, self.right_wing_CM])
        wings_tips = [self.wings_tips_left, self.wings_tips_right]
        for wing in range(2):
            point_0 = self.points_3D[:, 0 + wing*(self.num_points_per_wing), :]
            point_1 = self.points_3D[:, 1 + wing*(self.num_points_per_wing), :]
            wing_span = point_1 - point_0
            wing_span = wing_span / np.linalg.norm(wing_span, axis=1)[:, np.newaxis]
            wing_spans[:, wing, :] = wing_span
        left_wing_span = wing_spans[:, LEFT, :]
        right_wing_span = wing_spans[:, RIGHT, :]
        return left_wing_span, right_wing_span

    def get_spans_centers_to_tip(self):
        wing_spans = np.zeros((self.num_frames, 2, 3))
        wings_CM = np.array([self.left_wing_CM, self.right_wing_CM])
        wings_tips = [self.wings_tips_left, self.wings_tips_right]
        for wing in range(2):
            CMs = wings_CM[wing]
            tip = wings_tips[wing]
            wing_span = tip - CMs
            wing_span = wing_span / np.linalg.norm(wing_span, axis=1)[:, np.newaxis]
            wing_spans[:, wing, :] = wing_span
        left_wing_span = wing_spans[:, LEFT, :]
        right_wing_span = wing_spans[:, RIGHT, :]
        return left_wing_span, right_wing_span

    def get_z_body(self):
        z_body = np.cross(self.x_body, self.y_body, axis=-1)
        z_body = normalize(z_body, 'l2')
        return z_body

    def set_right_left(self):
        wing_CM_left = np.mean(self.points_3D[:, self.left_inds[:-1], :], axis=1)
        wing_CM_right = np.mean(self.points_3D[:, self.right_inds[:-1], :], axis=1)
        wings_vec = wing_CM_right - wing_CM_left
        wings_vec = wings_vec / np.linalg.norm(wings_vec, axis=-1)[:, np.newaxis]
        cross = np.cross(wings_vec, self.x_body, axis=-1)
        z = 2
        z_component = cross[:, z]
        mean_z_component = np.mean(z_component)
        need_flip = False
        if mean_z_component < 0:
            need_flip = True
        if need_flip:
            self.flip_right_left_points()

        to_flip = self.find_if_flip()
        if to_flip:
            self.flip_right_left_points()

    def flip_right_left_points(self):
        left = self.points_3D[:, self.left_inds, :]
        right = self.points_3D[:, self.right_inds, :]
        self.points_3D[:, self.left_inds, :] = right
        self.points_3D[:, self.right_inds, :] = left

    def load_points(self):
        return np.load(self.points_3D_path)

    def get_head_tail_vec(self, smooth=False):
        head_tail_vec = self.head_tail_points[:, 1, :] - self.head_tail_points[:, 0, :]
        head_tail_vec = normalize(head_tail_vec, axis=1, norm='l2')

        return head_tail_vec

    def get_wings_joints_vec(self, smooth=False):
        wings_joints_vec = self.wings_joints_points[:, LEFT] - self.wings_joints_points[:, RIGHT]
        wings_joints_vec = normalize(wings_joints_vec, axis=1, norm='l2')
        return wings_joints_vec

    @staticmethod
    def get_body_pitch(x_body):
        pitch = np.rad2deg(np.arcsin(x_body[:, 2]))
        return pitch

    @staticmethod
    def get_body_yaw(x_body):
        only_xy = normalize(x_body[:, :-1], axis=1, norm='l2')
        yaw = np.rad2deg(np.arctan2(only_xy[:, 1], only_xy[:, 0]))
        yaw = np.abs(np.unwrap(yaw + 180, period=360) - 180)
        return yaw

    @staticmethod
    def get_Rzy(phi, theta, x_body, yaw, pitch):
        num_frames = phi.shape[0]
        Rzy_all = []
        for i in range(num_frames):
            x_body_i = x_body[i]
            Rzy = FlightAnalysis.euler_rotation_matrix(np.deg2rad(yaw[i]), np.deg2rad(pitch[i]), psi_rad=0)
            Rzy_all.append(Rzy)
        Rzy_all = np.array(Rzy_all)
        return Rzy_all

    @staticmethod
    def get_body_roll(phi, theta, x_body, y_body, yaw, pitch, start, end):
        num_frames = len(x_body)
        Rzy_all = FlightAnalysis.get_Rzy(phi, theta, x_body, yaw, pitch)
        all_roll_angles = np.zeros(num_frames, )
        for frame in range(start, num_frames):
            Rzy = Rzy_all[frame]
            yb_frame = y_body[frame]
            rotated_yb_frame = Rzy @ yb_frame
            roll_frame = np.arctan2(rotated_yb_frame[2], rotated_yb_frame[1])
            # roll_frame = np.rad2deg(roll_frame)
            all_roll_angles[frame] = roll_frame
        all_roll_angles = np.array(all_roll_angles)
        all_roll_angles = all_roll_angles[start:end]
        unwrap_roll_angles = np.unwrap(all_roll_angles)
        unwrap_roll_angles = np.degrees(unwrap_roll_angles)
        # plt.plot(unwrap_roll_angles)
        # plt.show()
        all_roll_angles = np.zeros(num_frames, )
        all_roll_angles[start:end] = unwrap_roll_angles
        # roll_angle = np.rad2deg(np.arcsin(self.y_body[:, 2]))
        # roll_wings_joints = np.rad2deg(np.arcsin(self.wings_joints_vec[:, 2]))
        # plt.plot(all_roll_angles)
        # plt.show()
        return unwrap_roll_angles

    @staticmethod
    def get_dot(data, window_length=5, polyorder=2, sampling_rate=1):
        data_dot = savgol_filter(data, window_length, polyorder, deriv=1, delta=(1 / sampling_rate))
        # plt.plot(data_dot / 100)
        # plt.plot(data)
        # plt.axhline(y=0, color='gray', linestyle='--')
        # plt.show()
        return data_dot

    @staticmethod
    def calculate_angle(arr1, arr2):

        arr1[np.abs(arr1) < 1e-8] = 0
        arr2[np.abs(arr2) < 1e-8] = 0
        dot_product = FlightAnalysis.row_wize_dot(arr1, arr2)
        angles_rad = np.arccos(dot_product)
        angle_degrees = np.rad2deg(angles_rad)
        return angle_degrees

    @staticmethod
    def consistent_angle(arr1, arr2, stroke_normal):
        angles = []
        for i, (a, b, n) in enumerate(zip(arr1, arr2, stroke_normal)):
            cos_theta = np.dot(a, b)
            cos_theta = np.clip(cos_theta, -1, 1)  # Clipping to avoid numerical issues outside arccos range
            theta_rad = np.arccos(cos_theta)

            cross_prod = np.cross(a, b)
            if np.dot(cross_prod, n) < 0:  # Assuming the Z-axis as reference; change depending on your reference axis
                theta_rad = 2 * np.pi - theta_rad
            theta_deg = np.rad2deg(theta_rad)
            angles.append(theta_deg)  # Convert radians to degrees

        return np.array(angles)

    def get_wings_theta(self):
        wings_spans_left = self.left_wing_span
        wings_spans_right = self.right_wing_span
        stroke_normals = self.stroke_planes[:, :-1]
        theta_left = 90 - np.rad2deg(np.arccos(FlightAnalysis.row_wize_dot(wings_spans_left, stroke_normals)))
        theta_right = 90 - np.rad2deg(np.arccos(FlightAnalysis.row_wize_dot(wings_spans_right, stroke_normals)))
        return theta_left, theta_right

    def get_wings_phi(self):
        proj_left = FlightAnalysis.project_vector_to_plane(self.left_wing_span, self.stroke_planes)
        proj_right = FlightAnalysis.project_vector_to_plane(self.right_wing_span, self.stroke_planes)
        proj_xbody = FlightAnalysis.project_vector_to_plane(self.x_body, self.stroke_planes)

        projected_wings = np.array([proj_left, proj_right])
        phis = []
        for wing_num in range(2):
            proj = projected_wings[wing_num]
            stroke_plane_normal = self.stroke_planes[:, :-1]
            if wing_num == 1:
                stroke_plane_normal = -stroke_plane_normal
            phi = FlightAnalysis.consistent_angle(proj, proj_xbody, stroke_plane_normal)
            phi = 360 - phi
            # phi[:self.first_y_body_frame] = 0
            # phi[self.end_frame:] = 0
            phis.append(phi)

        phi_left, phi_right = phis

        # plt.plot(phi_left[self.start_frame:self.end_frame])
        # plt.plot(phi_right[self.start_frame:self.end_frame])
        # plt.show()

        return phi_left, phi_right

    def get_waving_frequency(self, data):
        refined_peaks, peak_values = self.get_peaks(data, self.first_y_body_frame, self.end_frame)
        distances = np.diff(refined_peaks)
        frequencies = (1 / distances) * SAMPLING_RATE
        average_distance = np.mean(distances)
        average_frequency = (1 / average_distance) * SAMPLING_RATE
        return refined_peaks, frequencies, average_frequency

    def get_phi_peaks(self, phi):
        max_peaks_inds, max_peak_values = self.get_peaks(phi, self.first_y_body_frame, self.end_frame, show=False, prominence=75)
        all_max_peaks = np.stack([max_peaks_inds, max_peak_values]).T
        min_peaks_inds, min_peak_values = self.get_peaks(-phi, self.first_y_body_frame, self.end_frame, show=False, prominence=75)
        min_peak_values = [-min_peak for min_peak in min_peak_values]
        return max_peak_values, max_peaks_inds, min_peak_values, min_peaks_inds

    @staticmethod
    def get_peaks(data, first_y_body_frame, end_frame, distance=50, prominence=100, show=False, window_size=7, height=None):
        peaks, _ = find_peaks(data, height=height, prominence=prominence, distance=distance)
        if show:
            plt.figure(figsize=(10, 6))
            x_values = np.arange(first_y_body_frame, end_frame)
            plt.plot(x_values, data[first_y_body_frame: end_frame], label='Data', marker='o')
        refined_peaks = []
        peak_values = []
        for peak in peaks:
            # Define the window around the peak
            start = max(0, peak - window_size // 2)
            end = min(len(data), peak + window_size // 2 + 1)

            # Fit a second-degree polynomial
            x = np.arange(start, end)
            y = data[start:end]
            coef = np.polyfit(x, y, 2)

            # Create a polynomial object
            p = Polynomial(coef[::-1])  # np.polyfit returns highest power first

            # Find the vertex of the parabola (the maximum point for a concave down parabola)
            vertex = -coef[1] / (2 * coef[0])

            # Check if vertex is within the window
            if start <= vertex <= end:
                refined_peak = vertex
            else:
                refined_peak = peak
            value = p(refined_peak)
            refined_peaks.append(refined_peak)
            peak_values.append(value)

            # Plotting the polynomial over the window
            if show:
                x_fit = np.linspace(start, end, 100)
                y_fit = p(x_fit)
                plt.plot(x_fit, y_fit, label=f'Fit around peak at {peak}', linestyle='--')
        if show:
            plt.scatter(peaks, data[peaks], color='red', label='Initial Peaks')
            plt.scatter(refined_peaks, data[np.round(refined_peaks).astype(int)], color='green', marker='x',
                        label='Refined Peaks')
            # plt.legend()
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Peak Detection and Polynomial Fit')
            plt.show()
        return refined_peaks, peak_values

    @staticmethod
    def project_vector_to_plane(Vs, Ps):
        stroke_normals = Ps[:, :-1]
        proj_N_V = FlightAnalysis.row_wize_dot(Vs, stroke_normals)[:, np.newaxis] * stroke_normals
        V_on_plane = Vs - proj_N_V
        # check if indeed on the plane:
        # check = FlightAnalysis.row_wize_dot(V_on_plane, stroke_normals)
        V_on_plane = V_on_plane / np.linalg.norm(V_on_plane, axis=1)[:, np.newaxis]
        return V_on_plane

    def get_stroke_planes(self):
        stroke_planes = np.full((self.num_frames, 4), np.nan)
        theta = np.pi / 4
        stroke_normal = self.rodrigues_rot(self.x_body[self.first_y_body_frame:self.end_frame],
                                           self.y_body[self.first_y_body_frame:self.end_frame], -theta)
        stroke_normal = normalize(stroke_normal, norm='l2')
        body_center = np.mean(self.head_tail_points[self.first_y_body_frame:self.end_frame], axis=1)
        d = - np.sum(np.multiply(stroke_normal, body_center), axis=1)
        stroke_planes[self.first_y_body_frame:self.end_frame] = np.column_stack((stroke_normal, d))
        return stroke_planes

    @staticmethod
    def euler_rotation_matrix(phi_rad, theta_rad, psi_rad):
        """
        Returns a 3D Euler rotation matrix given angles in radians.
        :param phi_rad: Rotation about the x-axis
        :param theta_rad: Rotation about the y-axis (negated as in the original function)
        :param psi_rad: Rotation about the z-axis
        :return: 3x3 rotation matrix
        """
        # Adjust theta angle (negate as in the original function)
        theta_rad = -theta_rad

        # Calculate trigonometric values for the angles
        cph = np.cos(phi_rad)
        sph = np.sin(phi_rad)
        cth = np.cos(theta_rad)
        sth = np.sin(theta_rad)
        cps = np.cos(psi_rad)
        sps = np.sin(psi_rad)

        # Define the matrix rows according to the MATLAB function's logic
        M1 = np.array([cth * cph, cth * sph, -sth])
        M2 = np.array([sps * sth * cph - cps * sph, sps * sth * sph + cps * cph, cth * sps])
        M3 = np.array([cps * sth * cph + sps * sph, cps * sth * sph - sps * cph, cth * cps])

        # Concatenate rows into a 3x3 matrix
        TrdRotmat = np.array([M1, M2, M3])

        return TrdRotmat

    @staticmethod
    def rodrigues_rot(V, K, theta):
        """
        Args:
            V:  the vector to rotate
            K: the axis of rotation
            theta: angle in radians
        Returns:

        """
        num_frames, ndims = V.shape[0], V.shape[1]
        V_rot = np.zeros_like(V)
        for frame in range(num_frames):
            vi = V[frame, :]
            ki = K[frame, :]
            vi_rot = np.cos(theta) * vi + np.cross(ki, vi) * np.sin(theta) + ki * np.dot(ki, vi) * (1 - np.cos(theta))
            V_rot[frame, :] = vi_rot
        return V_rot

    def get_planes(self, points_inds):
        planes = np.zeros((self.num_frames, 2, 4))
        for frame in range(self.num_frames):
            for wing in range(2):
                points = self.points_3D[frame, points_inds[wing], :]
                plane_P, error = self.fit_plane(points)
                if frame > 0:
                    prev_plane_P = planes[frame - 1, 0, :]
                    if np.dot(prev_plane_P, plane_P) < 0:
                        plane_P = -plane_P
                planes[frame, wing, :] = plane_P
        return planes

    @staticmethod
    def get_auto_correlation_axis_angle(x_body, y_body, z_body, start_frame, end_frame):
        # first_nonzero_index = np.argmax((y_body != 0).any(axis=1))
        # reversed_mat = np.flip(y_body, axis=0)
        # last_index_reversed = np.argmax((reversed_mat != 0).any(axis=1))
        # last_nonzero_index = y_body.shape[0] - 1 - last_index_reversed

        T = (end_frame - start_frame) // 2
        x_body, y_body, z_body = (x_body[start_frame:end_frame],
                                  y_body[start_frame:end_frame],
                                  z_body[start_frame:end_frame])
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            xb, yb, zb = x_body[:-df], y_body[:-df], z_body[:-df]
            Rs = np.stack([xb, yb, zb], axis=-1)

            xb_pair, yb_pair, zb_pair = x_body[df:], y_body[df:], z_body[df:]
            Rs_pair = np.stack([xb_pair, yb_pair, zb_pair], axis=-1)

            angels_radiance = np.array(
                [FlightAnalysis.get_rotation_axis_angle(Rs[i], Rs_pair[i]) for i in range(Rs.shape[0])])
            cosines = np.cos(angels_radiance)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def get_angular_velocities(x_body, y_body, z_body, start_frame, end_frame, sampling_rate=SAMPLING_RATE):
        num_frames = len(x_body)
        x_body, y_body, z_body = (x_body[start_frame:end_frame],
                                  y_body[start_frame:end_frame],
                                  z_body[start_frame:end_frame])
        Rs = np.stack([x_body, y_body, z_body], axis=-1)
        N = len(Rs)
        T = np.arange(N)
        dRdt = np.zeros_like(Rs)
        # find dRdt
        for i in range(3):
            for j in range(3):
                entry_ij = Rs[:, i, j]
                # spline = make_smoothing_spline(y=entry_ij, x=T)
                # vals = spline(T)
                # derivative = spline.derivative()(T)
                dRdt[:, i, j] = np.gradient(entry_ij)
        w_x, w_y, w_z = np.zeros((3, N))
        for frame in range(N):
            A_dot = dRdt[frame]
            A = Rs[frame]
            omega = A_dot @ A.T
            wx_frame = (omega[2, 1] - omega[1, 2]) / 2
            wy_frame = (omega[0, 2] - omega[2, 0]) / 2
            wz_frame = (omega[1, 0] - omega[0, 1]) / 2
            # print(wx_frame, wy_frame, wz_frame)
            w_x[frame] = np.rad2deg(wx_frame) * sampling_rate
            w_y[frame] = np.rad2deg(wy_frame) * sampling_rate
            w_z[frame] = np.rad2deg(wz_frame) * sampling_rate

        omega_lab = np.column_stack((w_x, w_y, w_z))
        omega_body = np.zeros_like(omega_lab)
        for frame in range(N):
            omega_lab_i = omega_lab[frame, :]
            R = Rs[frame]
            omega_body_i = R.T @ omega_lab_i
            omega_body[frame, :] = omega_body_i

        nan_frames = np.full((start_frame, 3), np.nan)
        omega_lab = np.concatenate((nan_frames, omega_lab), axis=0)
        omega_body = np.concatenate((nan_frames, omega_body), axis=0)

        nan_frames = np.full((num_frames - end_frame, 3), np.nan)
        omega_lab = np.concatenate((omega_lab, nan_frames), axis=0)
        omega_body = np.concatenate((omega_body, nan_frames), axis=0)

        angular_speed_body = np.linalg.norm(omega_body, axis=-1)
        angular_speed_lab = np.linalg.norm(omega_lab, axis=-1)

        return omega_lab, omega_body, angular_speed_lab, angular_speed_body

    @staticmethod
    def get_auto_correlation_x_body(x_body):
        T = len(x_body) // 2
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            x_bodies = x_body[:-df]
            x_bodies_pair = x_body[df:]
            cosines = FlightAnalysis.row_wize_dot(x_bodies, x_bodies_pair)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def get_rotation_axis_angle(vectors1, vectors2):
        # vectors1, vectors2 (3, 3) matrices of perpendicular unit vectors (each row is a vector)
        # Create rotation matrix
        R = np.dot(vectors2, vectors1.T)

        # Calculate rotation axis
        # _, v = np.linalg.eig(R)
        # axis = np.real(v[:, np.isclose(np.linalg.eigvals(R), 1)])

        # Calculate rotation angle
        angle_rad = FlightAnalysis.get_angle_of_rotation_around_rot_axis(R)
        return angle_rad

    @staticmethod
    def get_angle_of_rotation_around_rot_axis(R):
        angle_rad = np.arccos((np.trace(R) - 1) / 2)
        return angle_rad

    @staticmethod
    def fit_plane(points):

        # fit a plane
        pca = PCA(n_components=3)
        pca.fit(points)
        normal = pca.components_[-1]  # the z component of the data
        mean = pca.mean_
        d = - normal.T @ mean  # ax + by + cz + d = 0 -> ax + by + cz = -d
        plane_P = np.append(normal, d)

        # find the fit error
        error = np.mean(np.abs(points @ normal + d))
        return plane_P, error

    @staticmethod
    def auto_correlation_1D(angles, is_radians=False):
        """
        computes the auto-correlation between angles:
        C(dt) =  <cos[Theta(t) - theta(t + dt)]>
        """
        T = len(angles) // 2
        AC = np.zeros(T)
        AC[0] = 1
        if not is_radians:
            angles = np.deg2rad(angles)
        for df in range(1, T):
            d_theta = angles[df:] - angles[:-df]
            cosines = np.cos(d_theta)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def auto_correlation_3D(unit_vectors):
        T = len(unit_vectors) // 2
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            first_vecs = unit_vectors[df:]
            second_vecs = unit_vectors[:-df]
            cosines = FlightAnalysis.row_wize_dot(first_vecs, second_vecs)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def row_wize_dot(arr1, arr2):
        dot = np.sum(arr1 * arr2, axis=1)
        return dot

    @staticmethod
    def find_zero_crossings_up(data):
        # data is 1D
        # find when sign is changing from negative to positive
        return np.where(np.diff(np.sign(data)) > 0)[0]

    @staticmethod
    def remove_close_elements(data, threshold=60):
        diffs = np.abs(np.diff(data))
        indices = np.insert(diffs >= threshold, 0, True)
        return data[indices]

    def get_roni_y_body(self):
        idx4StrkPln = self.choose_span()
        if len(idx4StrkPln) == 1:
            idx4StrkPln = np.array([19, 85])
            print("only one index for the y body calculation")
        y_bodies = []
        for i, ind in enumerate(idx4StrkPln):
            indices_to_take = np.arange(ind - NUM_TIPS_EACH_SIZE_Y_BODY, ind + NUM_TIPS_EACH_SIZE_Y_BODY + 1)
            left = self.wings_tips_left[indices_to_take, :]  # []
            right = self.wings_tips_right[indices_to_take, :]
            points = np.concatenate((left, right), axis=0)
            wing_tips_plane = self.fit_plane(points)[0]
            plane_normal = wing_tips_plane[:-1]
            y_body = np.cross(plane_normal, self.x_body[ind])
            y_body = y_body / np.linalg.norm(y_body)
            left_span = self.left_wing_span[ind]
            if np.dot(y_body, left_span) < 0:
                y_body = - y_body
            y_bodies.append(y_body)
            # self.plot_plane_and_points(ind, wing_tips_plane, points, y_body)
            pass
        y_bodies = np.array(y_bodies)
        all_y_bodies = np.zeros_like(self.x_body)
        first_y_body_frame = np.min(idx4StrkPln)
        end = np.max(idx4StrkPln)
        x = np.arange(first_y_body_frame, end)
        kind = 'linear' if len(idx4StrkPln) == 2 else 'quadratic'
        assert len(idx4StrkPln) >= 2, "there must be more then 2 indices for the y body calculation"
        f1 = interp1d(idx4StrkPln, y_bodies[:, 0], kind=kind)
        f2 = interp1d(idx4StrkPln, y_bodies[:, 1], kind=kind)
        f3 = interp1d(idx4StrkPln, y_bodies[:, 2], kind=kind)
        Ybody_inter = np.vstack((f1(x), f2(x), f3(x))).T

        window_length = min(73 * 2, len(Ybody_inter))
        Ybody_inter = FlightAnalysis.savgol_smoothing(Ybody_inter[:, np.newaxis, :], lam=1, polyorder=1,
                                                      window_length=window_length, median_kernel=1)
        Ybody_inter = np.squeeze(Ybody_inter)
        Ybody_inter = normalize(Ybody_inter, axis=1, norm='l2')
        all_y_bodies[first_y_body_frame:end, :] = Ybody_inter
        # make sure that the all_y_bodies are (1) unit vectors and (2) perpendicular to x_body
        y_bodies_corrected = all_y_bodies - self.row_wize_dot(self.x_body, all_y_bodies).reshape(-1, 1) * self.x_body
        y_bodies_corrected = normalize(y_bodies_corrected, 'l2')

        # plt.plot(idx4StrkPln, y_bodies)
        # plt.plot(y_bodies_corrected)
        # plt.plot(self.wings_joints_vec)
        # plt.show()
        # y_bodies_corrected = self.wings_joints_vec
        # first_y_body_frame = 0
        # end = self.num_frames - 1
        return y_bodies_corrected, first_y_body_frame, end

    def choose_span(self):
        dotspanAx_wing1 = self.row_wize_dot(self.right_wing_span, self.x_body)
        dotspanAx_wing2 = self.row_wize_dot(self.left_wing_span, self.x_body)

        dotspanAx_wing2 = self.filloutliers(dotspanAx_wing2, 'pchip')
        dotspanAx_wing1 = self.filloutliers(dotspanAx_wing1, 'pchip')

        distSpans = np.degrees(
            np.arccos(np.clip(self.row_wize_dot(self.right_wing_span, self.left_wing_span), -1.0, 1.0)))
        angBodSp = np.degrees(
            np.arccos(np.clip(self.row_wize_dot(self.right_wing_span, self.x_body), -1.0, 1.0)))

        mean_strks = np.mean([dotspanAx_wing1, dotspanAx_wing2], axis=0)
        mean_strks = savgol_filter(mean_strks, window_length=5, polyorder=2)

        changeSgn = np.vstack((mean_strks < 0, mean_strks >= 0))

        FrbckStrk = self.find_up_down_strk(changeSgn, mean_strks, 0)
        idx4StrkPln = self.choose_grang_wing1_wing2(distSpans, FrbckStrk, 140, 10)

        diff_threshold = 60
        while np.any(np.diff(idx4StrkPln) < diff_threshold):
            idx4StrkPln = np.delete(idx4StrkPln, np.where(np.diff(idx4StrkPln) < diff_threshold)[0] + 1)

        idx4StrkPln = np.unique(idx4StrkPln)
        idx4StrkPln = idx4StrkPln[angBodSp[idx4StrkPln] > 70]

        idx4StrkPln = idx4StrkPln[
            (idx4StrkPln > NUM_TIPS_EACH_SIZE_Y_BODY) & (idx4StrkPln < (self.num_frames - NUM_TIPS_EACH_SIZE_Y_BODY))]

        return idx4StrkPln

    def choose_grang_wing1_wing2(self, distSpans, FrbckStrk, angleTH, Search_cell):
        idx4StrkPln = FrbckStrk[distSpans[FrbckStrk] > angleTH]

        for i in range(len(idx4StrkPln)):
            inds2ch = np.arange(idx4StrkPln[i] - Search_cell, idx4StrkPln[i] + Search_cell + 1)
            inds2ch = inds2ch[(inds2ch >= 0) & (inds2ch < len(distSpans))]
            idx4StrkPln[i] = inds2ch[np.argmax(distSpans[inds2ch])]

        return idx4StrkPln

    @staticmethod
    def filloutliers(data, method):
        # Simple implementation of outlier filling using interpolation
        if method == 'pchip':
            median = np.median(data)
            std_dev = np.std(data)
            threshold = 3 * std_dev
            outliers = np.abs(data - median) > threshold
            data[outliers] = np.interp(np.flatnonzero(outliers), np.flatnonzero(~outliers), data[~outliers])
            return data
        else:
            raise ValueError("Unsupported method")

    @staticmethod
    def find_up_down_strk(changeSgn, mean_strks, up_down_strk):
        """
        Find where the signs change (compared to 0 on the body axis).
        In this position, the wings are farthest apart.

        Parameters:
        changeSgn (ndarray): Array indicating sign changes.
        mean_strks (ndarray): Mean strokes array.
        up_down_strk (int): Value to compare for finding up and down strokes.

        Returns:
        ndarray: Indices of down strokes.
        """
        downstrk = np.where((changeSgn[0, :-1] + changeSgn[1, 1:]) == up_down_strk)[0]
        mean_val = np.vstack((mean_strks[downstrk + 1], mean_strks[downstrk]))
        indMin = np.argmin(np.abs(mean_val), axis=0)
        idx_vec = np.vstack((downstrk + 1, downstrk))
        idxdwnStrk = idx_vec[(indMin, np.arange(indMin.size))]
        return idxdwnStrk

    #################################################################################################################

    def plot_plane_and_points(self, ind, plane, points, y_body):
        size = 0.005
        cm = np.mean(self.head_tail_points[ind], axis=0)
        wing_points = points
        center = np.mean(points, axis=0)
        a, b, c, d = plane
        x = np.linspace(-size / 2, size / 2, 10) + center[0]
        y = np.linspace(-size / 2, size / 2, 10) + center[1]
        xx, yy = np.meshgrid(x, y)
        zz = (-a * xx - b * yy - d) / c
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot surface
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, label='Plane')
        # plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # plot center of mass
        ax.scatter(*cm, color="r", s=100)
        # plot head and tail
        ax.scatter(self.head_tail_points[ind, :, 0], self.head_tail_points[ind, :, 1], self.head_tail_points[ind, :, 2])
        # Plot the vectors
        ax.quiver(*cm, *(y_body * 0.001), color="b", alpha=.8)
        # ax.quiver(*cm, *(np.array([a, b, c]) * 0.001), color="g", alpha=.8)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    @staticmethod
    def fit_a_line_to_points(points):
        ##############################
        pca = PCA(n_components=3)
        pca.fit(points)
        best_fit_direction = pca.components_[0]
        print(pca.explained_variance_ratio_)
        ###############################
        print(FlightAnalysis.calculate_straightness(points))

    @staticmethod
    def calculate_straightness(trajectory):
        total_distance = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1)))
        euclidean_distance = np.sqrt(np.sum((trajectory[-1] - trajectory[0]) ** 2))
        return euclidean_distance / total_distance

    @staticmethod
    def sort_trajectories(trajectories):
        straightness_values = [FlightAnalysis.calculate_straightness(trajectory) for trajectory in trajectories]
        sorted_indices = np.argsort(straightness_values)[::-1]  # Sort in descending order
        sorted_trajectories = [trajectories[i] for i in sorted_indices]
        return sorted_trajectories

    @staticmethod
    def enforce_3D_consistency(points_3D, right_inds, left_inds):
        num_frames = points_3D.shape[0]
        right_points = np.zeros((num_frames, len(right_inds), 3))
        left_points = np.zeros((num_frames, len(left_inds), 3))
        # initialize
        right_points[0, ...] = points_3D[0, right_inds]
        left_points[0, ...] = points_3D[0, left_inds]

        for frame in range(1, num_frames):
            cur_left_points = points_3D[frame, left_inds, :]
            cur_right_points = points_3D[frame, right_inds, :]

            prev_left_points = left_points[frame - 1, :]
            prev_right_points = right_points[frame - 1, :]

            l2l_dist = np.linalg.norm(cur_left_points - prev_left_points)
            r2r_dist = np.linalg.norm(cur_right_points - prev_right_points)
            r2l_dist = np.linalg.norm(cur_right_points - prev_left_points)
            l2r_dist = np.linalg.norm(cur_left_points - prev_right_points)
            do_switch = l2l_dist + r2r_dist > r2l_dist + l2r_dist

            if do_switch:
                # print(f'switch occuered in frame {frame}')
                right_points[frame] = cur_left_points
                left_points[frame] = cur_right_points
            else:
                right_points[frame] = cur_right_points
                left_points[frame] = cur_left_points
        points_3D[:, right_inds, :] = right_points
        points_3D[:, left_inds, :] = left_points
        return points_3D

    @staticmethod
    def euler_rotation_matrix(phi_rad, theta_rad, psi_rad):
        """
        Returns the Euler rotation matrix for the given angles in radians.

        Parameters:
        phi_rad (float): Rotation angle around the x-axis (in radians).
        theta_rad (float): Rotation angle around the y-axis (in radians).
        psi_rad (float): Rotation angle around the z-axis (in radians).

        Returns:
        np.ndarray: The 3x3 Euler rotation matrix.
        """
        theta_rad = -theta_rad

        cph = np.cos(phi_rad)
        sph = np.sin(phi_rad)
        cth = np.cos(theta_rad)
        sth = np.sin(theta_rad)
        cps = np.cos(psi_rad)
        sps = np.sin(psi_rad)

        M = np.array([
            [cth * cph, cth * sph, -sth],
            [sps * sth * cph - cps * sph, sps * sth * sph + cps * cph, cth * sps],
            [cps * sth * cph + sps * sph, cps * sth * sph - sps * cph, cth * cps]
        ])

        return M



def save_movies_data_to_hdf5(base_path, output_hdf5_path, smooth=True, one_h5_for_all=False):
    # Ensure the directory exists
    if not os.path.isdir(base_path):
        print(f"Directory {base_path} does not exist")
        return

    if os.path.isfile(output_hdf5_path) and one_h5_for_all:
        print(f"File {output_hdf5_path} exists")
        return

    # Filter out directories not starting with 'mov'
    movies = [dir for dir in os.listdir(base_path) if dir.startswith('mov') if dir != 'mov3']
    # Create an HDF5 file for each movie
    movies = ['mov3']
    for movie in movies:
        print(f"now proccessing {movie}", flush=True)
        movie_dir = os.path.join(base_path, movie)
        points_3D_path = os.path.join(movie_dir,
                                      'points_3D_smoothed_ensemble_best_method.npy') if smooth else os.path.join(
            movie_dir,
            'points_3D_ensemble_best_method.npy')

        if os.path.isfile(points_3D_path):
            try:
                create_movie_analysis_h5(movie, movie_dir, points_3D_path, smooth)
            except Exception as e:
                print(f"Exception occurred while processing {movie}: {e}", flush=True)
        else:
            print(f"Missing data for {movie}", flush=True)

    print(
        f"All data saved to {output_hdf5_path}" if one_h5_for_all else "All data saved to individual movie HDF5 files")


def create_movie_analysis_h5(movie, movie_dir, points_3D_path, smooth, analysis_object=None):
    if analysis_object is None:
        FA = FlightAnalysis(points_3D_path, create_html=True)  # Assuming FlightAnalysis is properly defined
    else:
        FA = analysis_object
    name = f'{movie}_analysis_smoothed.h5' if smooth else f'{movie}_analysis.h5'
    # Define the output HDF5 file path for the movie
    movie_hdf5_path = os.path.join(movie_dir, name)
    # Create the HDF5 file for this movie
    with h5py.File(movie_hdf5_path, 'w') as hdf:
        # Store attributes as datasets within the group
        for attr_name, attr_value in FA.__dict__.items():
            if attr_name in ["left_half_wingbits", "right_half_wingbits",
                             "left_full_wingbits", "right_full_wingbits"]:
                wing_bit_part = attr_value
                kind = "half" if "half" in attr_name else "full"
                group = hdf.create_group(attr_name)  # Create a group for half_wingbits
                for i, obj in enumerate(wing_bit_part):
                    obj_group = group.create_group(f'{kind}_wingbit_{i}')
                    try:
                        for sub_attr_name, sub_attr_value in obj.__dict__.items():
                            # Create a dataset inside the group for each attribute of the object
                            obj_group.create_dataset(sub_attr_name, data=sub_attr_value)
                    except:
                        print(f'{kind}_wingbit_{i} does not exsist')
            elif attr_name in ["full_body_wing_bits"]:
                group = hdf.create_group(attr_name)
                for i, obj in enumerate(attr_value):
                    obj_group = group.create_group(f'full_body_wingbit_{i}')
                    try:
                        for sub_attr_name, sub_attr_value in obj.__dict__.items():
                            # Create a dataset inside the group for each attribute of the object
                            obj_group.create_dataset(sub_attr_name, data=sub_attr_value)
                    except:
                        print(f'full_body_wingbit_{i} does not exsist')
            else:
                if hasattr(attr_value, '__len__'):
                    # print(attr_name) # Check for array-like objects or lists
                    hdf.create_dataset(attr_name, data=attr_value)
                elif isinstance(attr_value, (int, float)):
                    # Check for int or float attributes
                    hdf.create_dataset(attr_name, data=attr_value)
    print(f"Data saved for {movie} in {movie_hdf5_path}")
    Visualizer.plot_all_body_data(movie_hdf5_path)
    return movie_hdf5_path, FA


def create_one_movie_analisys():
    num = 101
    movie = f"mov{num}"
    movie_dir = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{num}"
    points_3D_path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{num}\points_3D_smoothed_ensemble_best_method.npy"
    # movie = "mov78"
    # movie_dir = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78"
    # points_3D_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\points_3D_smoothed_ensemble_best_method.npy"
    # smooth = True
    create_movie_analysis_h5(movie, movie_dir, points_3D_path, True)



def analize_all_movies():
    base_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms"
    save_movies_data_to_hdf5(base_path, output_hdf5_path="", smooth=True, one_h5_for_all=False)
    base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
    # save_movies_data_to_hdf5(base_path, output_hdf5_path="", smooth=True, one_h5_for_all=False)
    base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
    # save_movies_data_to_hdf5(base_path, output_hdf5_path="", smooth=True, one_h5_for_all=False)


if __name__ == '__main__':
    path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
    path_intact = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms"
    # points_3D_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov53\points_3D_smoothed_ensemble_best_method.npy"
    points_3D_path = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight\mov376\points_3D_smoothed_ensemble_best_method.npy"
    # points_3D_path = r"C:\Users\amita\OneDrive\Desktop\roni bad movies\points_3D_ensemble_best_method.npy"
    # points_3D_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/test movies/mov6/points_3D_smoothed_ensemble_best_method.npy"
    FlightAnalysis(points_3D_path=points_3D_path,
                   create_html=False,
                   create_mp4=False,
                   create_h5=True,
                   find_auto_correlation=True)
    # analize_all_movies()



