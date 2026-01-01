import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.widgets import Slider, Button
from matplotlib import colors
import sklearn
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from prediction_code_lior.extract_flight_data import FlightAnalysis
from scipy.interpolate import CubicSpline


def row_wize_dot(arr1, arr2):
    dot = np.sum(arr1 * arr2, axis=1)
    return dot


def do_savgol_filter(array, window_size, order):
    if array.ndim == 1:
        array = array[:, np.newaxis]
    smoothed_array = np.zeros_like(array)
    for axis in range(array.shape[-1]):
        smoothed_array[:, axis] = savgol_filter(np.copy(array[:, axis]), window_size, order)
    return smoothed_array.squeeze()


class DataComparison:
    def __init__(self, mov_num, path_my_data, path_roni_data, smoothed, smooth_like_roni=False):
        self.frame_rate = 16000
        self.mov_num = mov_num
        if smoothed:
            self.path_my_data = os.path.join(path_my_data, "all_movies_data_smoothed.h5")
            self.path_roni_data = os.path.join(path_roni_data, "smoothed")
        else:
            self.path_my_data = os.path.join(path_my_data, "all_movies_data_not_smoothed.h5")
            self.path_roni_data = os.path.join(path_roni_data, "not smoothed")
        self.my_data = None
        self.roni_data = None
        self.smooth_like_roni = smooth_like_roni
        self.load_my_data()
        self.load_roni_data()

    def load_my_data(self):
        with h5py.File(self.path_my_data, 'r') as hdf:
            group = hdf[f"mov{self.mov_num}"]
            self.my_data = {
                'points_3D': group['points_3D'][:],
                'x_body': group['x_body'][:],
                'y_body': group['y_body'][:],
                'z_body': group['z_body'][:],
                'frames': np.arange(len(group['x_body'][:])),
                'yaw_angle': group['yaw_angle'][:].squeeze() + 360,
                'pitch_angle': group['pitch_angle'][:].squeeze(),
                'roll_angle': group['roll_angle'][:].squeeze(),
                'center_of_mass': group['center_of_mass'][:],
                'CM_velocity': group['CM_velocity'][:],
                'left_wing_span': group['left_wing_span'][:],
                'right_wing_span': group['right_wing_span'][:],
                'left_wing_CM': group['left_wing_CM'][:],
                'right_wing_CM': group['right_wing_CM'][:],
                'stroke_planes': group['stroke_planes'][:, :-1],
                'wings_phi_left': group['wings_phi_left'][:].squeeze(),
                'wings_theta_left': group['wings_theta_left'][:].squeeze(),
                'wings_psi_left': group['wings_psi_left'][:].squeeze() + 180,
                'wings_phi_right': group['wings_phi_right'][:].squeeze(),
                'wings_theta_right': group['wings_theta_right'][:].squeeze(),
                'wings_psi_right': group['wings_psi_right'][:].squeeze() + 0,
                'left_wing_chord': group['left_wing_chord'][:],
                'right_wing_chord': group['right_wing_chord'][:],
                'wings_tips_left': group['wings_tips_left'][:],
                'wings_tips_right': group['wings_tips_right'][:],
                'all_2_planes': group['all_2_planes'][:],
            }
            if self.smooth_like_roni:
                self.my_data['center_of_mass'] = do_savgol_filter(self.my_data['center_of_mass'], window_size=73*3, order=3)
                # body angles
                self.my_data['yaw_angle'] = do_savgol_filter(self.my_data['yaw_angle'], window_size=15, order=2)
                self.my_data['pitch_angle'] = do_savgol_filter(self.my_data['pitch_angle'], window_size=15, order=2)
                self.my_data['roll_angle'] = do_savgol_filter(self.my_data['roll_angle'], window_size=15, order=2)
                # wings
                self.my_data['wings_phi_left'] = do_savgol_filter(self.my_data['wings_phi_left'], window_size=15, order=2)
                self.my_data['wings_psi_left'] = do_savgol_filter(self.my_data['wings_psi_left'], window_size=15, order=2)
                self.my_data['wings_theta_left'] = do_savgol_filter(self.my_data['wings_theta_left'], window_size=15, order=2)

                self.my_data['wings_phi_right'] = do_savgol_filter(self.my_data['wings_phi_right'], window_size=15,
                                                                  order=2)
                self.my_data['wings_psi_right'] = do_savgol_filter(self.my_data['wings_psi_right'], window_size=15,
                                                                  order=2)
                self.my_data['wings_theta_right'] = do_savgol_filter(self.my_data['wings_theta_right'], window_size=15,
                                                                    order=2)


    def read_specific_columns(self, csv_path, column_names):
        return pd.read_csv(csv_path, usecols=column_names).to_numpy()

    def load_roni_data(self):
        base_path = os.path.join(self.path_roni_data, f"mov{self.mov_num}")
        self.roni_data = {
            'frames': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['frames']).astype(int).squeeze() - 1,
            'x_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['X_x_body', 'X_y_body', 'X_z_body']),
            'y_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['Y_x_body', 'Y_y_body', 'Y_z_body']),
            'z_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['Z_x_body', 'Z_y_body', 'Z_z_body']),
            'yaw_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['yaw_body']).squeeze(),
            'pitch_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['pitch_body']).squeeze(),
            'roll_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['roll_body']).squeeze(),
            'center_of_mass': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['CM_real_x_body', 'CM_real_y_body', 'CM_real_z_body']),
            # 'CM_velocity': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['CM_real_x_body_dot_dot_smth', 'CM_real_y_body_dot_dot_smth', 'CM_real_z_body_dot_dot_smth']),
            'left_wing_span': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['span_x_lw', 'span_y_lw', 'span_z_lw']),
            'right_wing_span': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['span_x_rw', 'span_y_rw', 'span_z_rw']),
            'stroke_planes': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['strkPlan_x_body', 'strkPlan_y_body', 'strkPlan_z_body']),
            'phi_left': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['phi_lw']).squeeze(),
            'theta_left': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['theta_lw']).squeeze(),
            'psi_left': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['psi_lw']).squeeze(),
            'phi_right': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['phi_rw']).squeeze(),
            'theta_right': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['theta_rw']).squeeze(),
            'psi_right': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['psi_rw']).squeeze(),
            'left_wing_chord': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['chord_x_lw', 'chord_y_lw', 'chord_z_lw']),
            'right_wing_chord': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['chord_x_rw', 'chord_y_rw', 'chord_z_rw']),
            # 'left_wing_tips': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['tip_x_lw', 'tip_y_lw', 'tip_z_lw']),
            # 'right_wing_tips': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['tip_x_rw', 'tip_y_rw', 'tip_z_rw'])
        }

    def plot_vectors(self, my_data, roni_data, vector_name):
        labels = ['X', 'Y', 'Z']
        convert_to_ms = self.frame_rate / 1000
        fig = go.Figure()
        for i, label in enumerate(labels):
            # fig.add_trace(go.Scatter(
            #     x=roni_data['frames'][:] ,
            #     y=roni_data[:, i],
            #     mode='lines',
            #     name=f'Roni {label} {vector_name}'
            # ))
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.my_data)) ,
                y=my_data[:, i],
                mode='lines+markers',
                name=f'My {label} {vector_name}',
                # line=dict(dash='dash')
            ))
        fig.update_layout(
            title=f'{vector_name} Vectors',
            xaxis_title='Milliseconds',
            yaxis_title=f'{vector_name} Coordinate',
            legend_title='Legend'
        )
        html_file_name = f'{vector_name}_vectors.html'
        fig.write_html(html_file_name)  # Save the figure as HTML file
        print(f'Saved: {html_file_name}')
        fig.show()

    def compare_body_vectors(self):
        self.plot_vectors(self.my_data['x_body'], self.roni_data['x_body'], "X Body")
        self.plot_vectors(self.my_data['y_body'], self.roni_data['y_body'], "Y Body")
        self.plot_vectors(self.my_data['z_body'], self.roni_data['z_body'], "Z Body")

    def compare_body_angles(self):
        angles = ['yaw', 'pitch', 'roll']
        convert_to_ms = self.frame_rate / 1000
        for angle in angles:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.my_data['frames'],
                y=self.my_data[f'{angle}_angle'],
                mode='lines',
                name=f'My {angle.capitalize()}',
                # line=dict(dash='dash')
            ))
            fig.update_layout(
                title=f'{angle.capitalize()} Body Angle',
                xaxis_title='Milliseconds',
                yaxis_title=f'{angle.capitalize()} Angle (degrees)',
                legend_title='Legend'
            )
            fig.write_html(f'{angle}_body_angle.html')

    def plot_body_pitch_phi_right_phi_left(self):
        # Assuming frame_rate and frames are defined and appropriately calculated in your class
        convert_to_ms = self.frame_rate / 1000

        # Create a figure with subplots
        fig = go.Figure()

        # Plot Pitch Body Angle
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'],
            y=self.my_data['pitch_angle'],
            mode='lines',
            name='My Pitch Body Angle',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_left'],
            mode='lines',
            name='My Left Wing Phi Angle',
            line=dict(color='red')
        ))

        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_right'],
            mode='lines',
            name='My Right Wing Phi Angle',
            line=dict(color='green')
        ))

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Body Pitch and Wing Phi Angles',
            xaxis_title='Milliseconds',
            yaxis_title='Angle (degrees)',
            legend_title='Legend'
        )

        # Save the plot as an HTML file
        html_file_name = 'body_pitch_phi_wings.html'
        fig.write_html(html_file_name)
        print(f'Saved: {html_file_name}')

        # Display the plot
        # fig.show()

    def plot_body_roll_phi_right_phi_left(self):
        # Assuming frame_rate and frames are defined and appropriately calculated in your class
        convert_to_ms = self.frame_rate / 1000

        # Create a figure with subplots
        fig = go.Figure()

        # Plot Phi Angle for Left Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_left'],
            mode='lines',
            name='My Left Wing Phi Angle',
            line=dict(color='red')
        ))

        # Plot Phi Angle for Right Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_right'],
            mode='lines',
            name='My Right Wing Phi Angle',
            line=dict(color='green')
        ))

        # Plot Roll Body Angle
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['roll_angle'],
            mode='lines',
            name='My Roll Body Angle',
            line=dict(color='blue')
        ))

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Wing Phi Angles and Body Roll',
            xaxis_title='Milliseconds',
            yaxis_title='Angle (degrees)',
            legend_title='Legend'
        )

        # Save the plot as an HTML file
        html_file_name = 'body_roll_phi_wings.html'
        fig.write_html(html_file_name)
        print(f'Saved: {html_file_name}')

    def plot_body_pitch_theta_left_theta_right(self):
        # Assuming frame_rate and frames are defined and appropriately calculated in your class
        convert_to_ms = self.frame_rate / 1000

        # Create a figure with subplots
        fig = go.Figure()

        # Plot Body Pitch Angle
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['pitch_angle'],
            mode='lines',
            name='My Body Pitch Angle',
            line=dict(color='blue')
        ))

        # Plot Theta Angle for Left Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_theta_left'],
            mode='lines',
            name='My Left Wing Theta Angle',
            line=dict(color='red')
        ))

        # Plot Theta Angle for Right Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_theta_right'],
            mode='lines',
            name='My Right Wing Theta Angle',
            line=dict(color='green')
        ))

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Body Pitch and Wing Theta Angles',
            xaxis_title='Milliseconds',
            yaxis_title='Angle (degrees)',
            legend_title='Legend'
        )

        # Save the plot as an HTML file
        html_file_name = 'body_pitch_theta_wings.html'
        fig.write_html(html_file_name)
        print(f'Saved: {html_file_name}')


    def compare_CM(self):
        self.plot_vectors(self.my_data['center_of_mass'], self.roni_data['center_of_mass'], "Center of Mass")

    def compare_CM_velocity(self):
        self.plot_vectors(self.my_data['CM_velocity'], self.roni_data['CM_velocity'], "Center of Mass Velocity")

    def compare_stroke_planes(self):
        self.plot_vectors(self.my_data['stroke_planes'], self.roni_data['stroke_planes'], "Stroke Planes")

    def compare_wings_span(self, wing='left'):
        if wing == 'left':
            self.plot_vectors(self.my_data['left_wing_span'], self.roni_data['left_wing_span'], "Left Wing Span")
        else:
            self.plot_vectors(self.my_data['right_wing_span'], self.roni_data['right_wing_span'], "Right Wing Span")

    def compare_wings_angles(self, wing='left'):
        angles = ['phi', 'theta', 'psi']
        for angle in angles:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.roni_data['frames'],
                y=self.roni_data[f'{angle}_{wing}'],
                mode='lines',
                name=f'Roni {angle.capitalize()} {wing.capitalize()} Wing'
            ))
            fig.add_trace(go.Scatter(
                x=self.my_data['frames'],
                y=self.my_data[f'wings_{angle}_{wing}'],
                mode='lines',
                name=f'My {angle.capitalize()} {wing.capitalize()} Wing',
                # line=dict(dash='dash')
            ))
            fig.update_layout(
                title=f'{angle.capitalize()} {wing.capitalize()} Wing Angle',
                xaxis_title='Frame',
                yaxis_title=f'{angle.capitalize()} Angle (degrees)',
                legend_title='Legend'
            )
            html_file_name = f'{angle}_{wing}_wing_angle.html'
            fig.write_html(html_file_name)  # Save the figure as HTML file
            print(f'Saved: {html_file_name}')
            # fig.show()

    def theta_vs_phi(self, wing='left'):
        # for angle in ['phi', 'theta', 'psi']:
        plt.figure()
        add = 0
        start = 280 + add
        end = 489 + add
        plt.plot(self.my_data[f'wings_phi_{wing}'][start:end], self.my_data[f'wings_psi_{wing}'][start:end], linestyle='--', label=f'')
        plt.plot(self.roni_data[f'phi_{wing}'][start:end], self.roni_data[f'psi_{wing}'][start:end],
                 linestyle='-', label=f'')
        plt.xlabel('phi')
        plt.ylabel(f'theta Angle (degrees)')
        plt.title(f'theta_vs_phi Wing Angle')
        plt.legend()
        plt.axis('equal')
        plt.show()

    def compare_wings_chords(self, wing='left'):
        if wing == 'left':
            self.plot_vectors(self.my_data['left_wing_chord'], self.roni_data['left_wing_chord'], "Left Wing Chord")
        else:
            self.plot_vectors(self.my_data['right_wing_chord'], self.roni_data['right_wing_chord'], "Right Wing Chord")

    def compare_wings_tip(self, wing='left'):
        if wing == 'left':
            self.plot_vectors(self.my_data['wings_tips_left'], self.roni_data['left_wing_tips'], "Left Wing Tips")
        else:
            self.plot_vectors(self.my_data['wings_tips_right'], self.roni_data['right_wing_tips'], "Right Wing Tips")

    @staticmethod
    def row_wize_normalization(array):
        valid_rows = ~np.isnan(array).any(axis=1)
        valid_array = array[valid_rows]
        l2_norms = np.linalg.norm(valid_array, axis=1, keepdims=True)
        l2_norms[l2_norms == 0] = 1
        normalized_valid_array = valid_array / l2_norms
        normalized_array = np.full_like(array, np.nan)
        normalized_array[valid_rows] = normalized_valid_array
        return normalized_array

    def visualize_3D_comparison(self):
        # Assuming points is your (N, M, 3) array
        roni_frames = self.roni_data['frames'][:]
        points = self.my_data['points_3D'][:]
        num_frames = len(points)

        # stroke planes
        my_stroke_planes = self.my_data['stroke_planes'][:]

        # center of masss
        my_CM = self.my_data['center_of_mass'][:]
        attribute = 'center_of_mass'
        roni_CM = self.get_ronis_array(attribute, my_CM, roni_frames)

        # body vectors
        my_x_body = self.my_data['x_body'][:]
        my_y_body = self.my_data['y_body'][:]
        my_z_body = self.my_data['z_body'][:]

        check = row_wize_dot(my_z_body, my_z_body)
        mean = np.nanmean(check)

        # roni's body vectors
        roni_x_body = self.get_ronis_array('x_body', my_x_body, roni_frames)
        roni_y_body = self.get_ronis_array('y_body', my_y_body, roni_frames)
        roni_z_body = self.get_ronis_array('z_body', my_z_body, roni_frames)

        # wing vectors
        my_left_wing_span = self.my_data['left_wing_span'][:]
        my_right_wing_span = self.my_data['right_wing_span'][:]
        my_left_wing_chord = self.my_data['left_wing_chord'][:]
        my_right_wing_chord = self.my_data['right_wing_chord'][:]
        my_left_wing_normal = DataComparison.row_wize_normalization(np.cross(my_left_wing_chord, my_left_wing_span, axis=1))
        my_right_wing_normal = DataComparison.row_wize_normalization(np.cross(my_right_wing_chord, my_right_wing_span, axis=1))

        # roni's wing vectors
        roni_left_wing_span = self.get_ronis_array('left_wing_span', my_left_wing_span, roni_frames)
        roni_right_wing_span = self.get_ronis_array('right_wing_span', my_right_wing_span, roni_frames)
        roni_left_wing_chord = self.get_ronis_array('left_wing_chord', my_left_wing_chord, roni_frames)
        roni_right_wing_chord = self.get_ronis_array('right_wing_chord', my_right_wing_chord, roni_frames)
        roni_left_wing_normal = DataComparison.row_wize_normalization(np.cross(roni_left_wing_chord, roni_left_wing_span, axis=1))
        roni_right_wing_normal = DataComparison.row_wize_normalization(np.cross(roni_right_wing_chord, roni_right_wing_span, axis=1))

        # wing tips
        my_left_wing_tip = self.my_data['wings_tips_left'][:]
        my_right_wing_tip = self.my_data['wings_tips_right'][:]

        # wings cm
        my_left_wing_CM = self.my_data['left_wing_CM'][:]
        my_right_wing_CM = self.my_data['right_wing_CM'][:]

        # roni's cm
        left_dist_from_tip_to_cm = np.linalg.norm(my_left_wing_tip - my_left_wing_CM, axis=1)
        roni_left_wing_CM = my_left_wing_tip - left_dist_from_tip_to_cm[:, np.newaxis] * roni_left_wing_span

        right_dist_from_tip_to_cm = np.linalg.norm(my_right_wing_tip - my_right_wing_CM, axis=1)
        roni_right_wing_CM = my_right_wing_tip - right_dist_from_tip_to_cm[:, np.newaxis] * roni_right_wing_span

        points = self.my_data['points_3D'][:1000]
        # Calculate the limits of the plot
        x_min = np.nanmin(points[:, :, 0])
        y_min = np.nanmin(points[:, :, 1])
        z_min = np.nanmin(points[:, :, 2])

        x_max = np.nanmax(points[:, :, 0])
        y_max = np.nanmax(points[:, :, 1])
        z_max = np.nanmax(points[:, :, 2])

        # Create a color array
        my_points_to_show = np.array([my_left_wing_CM, my_right_wing_CM, my_CM, my_left_wing_tip, my_right_wing_tip])
        roni_points_to_show = np.array([roni_left_wing_CM, roni_right_wing_CM, roni_CM])
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       # (7, 15),
                       (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def add_quiver_axes(ax, origin, x_vec, y_vec, z_vec, scale=0.001, labels=None,  color='r'):
            x, y, z = origin
            if x_vec is not None:
                ax.quiver(x, y, z, x_vec[0], x_vec[1], x_vec[2], length=scale, color=color)
            if y_vec is not None:
                ax.quiver(x, y, z, y_vec[0], y_vec[1], y_vec[2], length=scale, color=color)
            if z_vec is not None:
                ax.quiver(x, y, z, z_vec[0], z_vec[1], z_vec[2], length=scale, color=color)

            # Add optional labels if provided
            if labels:
                if x_vec is not None:
                    ax.text(x + x_vec[0] * scale, y + x_vec[1] * scale, z + x_vec[2] * scale, labels[0], color=color)
                if y_vec is not None:
                    ax.text(x + y_vec[0] * scale, y + y_vec[1] * scale, z + y_vec[2] * scale, labels[1], color=color)
                if z_vec is not None:
                    ax.text(x + z_vec[0] * scale, y + z_vec[1] * scale, z + z_vec[2] * scale, labels[2], color=color)

        def plot_plane(ax, center, normal, size=0.005, color='g', alpha=0.25):
            d = size / 2

            # Create two orthogonal vectors on the plane
            if np.abs(normal[2]) > 0.99:
                u = np.array([1.0, 0.0, 0.0])
            else:
                u = np.cross(normal, [0.0, 0.0, 1.0])
            u = u / np.linalg.norm(u)

            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)

            # Calculate the corners of the plane
            corners = np.array([
                center + d * (u + v),
                center + d * (u - v),
                center + d * (-u - v),
                center + d * (-u + v),
            ])

            # Create the polygon for the plane
            vertices = [corners]
            plane = Poly3DCollection(vertices, color=color, alpha=alpha)
            ax.add_collection3d(plane)

        def update(val, plot_head_tail_points=False, plot_all_my_points=True, show_roni=True, show_me=True):
            ax.cla()  # Clear current axes
            frame = int(slider.val)

            if plot_all_my_points:
                for i in range(num_points):
                    if i not in [7, 15]:
                        ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
                for i, j in connections:
                    ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')

            # Plot points
            if plot_head_tail_points:
                ax.scatter(points[frame, -1, 0], points[frame, -1, 1], points[frame, -1, 2], c=color_array[0])
                ax.scatter(points[frame, -2, 0], points[frame, -2, 1], points[frame, -2, 2], c=color_array[0])
                ax.plot(points[frame, [-1, -2], 0], points[frame, [-1, -2], 1], points[frame, [-1, -2], 2], c='k')

            # Plot wing tips
            ax.scatter(my_left_wing_tip[frame, 0], my_left_wing_tip[frame, 1], my_left_wing_tip[frame, 2],
                       c=color_array[0])
            ax.scatter(my_right_wing_tip[frame, 0], my_right_wing_tip[frame, 1], my_right_wing_tip[frame, 2],
                       c=color_array[0])

            # Plot the center of mass
            my_cm_x, my_cm_y, my_cm_z = my_CM[frame]
            ax.scatter(my_cm_x, my_cm_y, my_cm_z, c=color_array[0])

            # Add the stroke plane
            plot_plane(ax, my_CM[frame], my_stroke_planes[frame])

            if show_me:
                add_quiver_axes(ax, (my_cm_x, my_cm_y, my_cm_z), my_x_body[frame], my_y_body[frame], my_z_body[frame],
                                color='r', labels=['Xb', 'Yb', 'Zb'])
                add_quiver_axes(ax, my_left_wing_CM[frame], my_left_wing_span[frame], my_left_wing_chord[frame], None,
                                color='r', labels=['Span', 'Chord'])
                add_quiver_axes(ax, my_right_wing_CM[frame], my_right_wing_span[frame], my_right_wing_chord[frame], None,
                                color='r', labels=['Span', 'Chord'])
            if show_roni:
                # Plot Roni's center of mass
                roni_cm_x, roni_cm_y, roni_cm_z = roni_CM[frame]
                ax.scatter(roni_cm_x, roni_cm_y, roni_cm_z, c=color_array[8])
                add_quiver_axes(ax, (roni_cm_x, roni_cm_y, roni_cm_z), roni_x_body[frame], roni_y_body[frame],
                                roni_z_body[frame], color='b', labels=['Xb', 'Yb', 'Zb'])
                add_quiver_axes(ax, roni_left_wing_CM[frame], roni_left_wing_span[frame], roni_left_wing_chord[frame], None,
                                color='b', labels=['Span', 'Chord'])
                add_quiver_axes(ax, roni_right_wing_CM[frame], roni_right_wing_span[frame], roni_right_wing_chord[frame],
                                None, color='b', labels=['Span', 'Chord'])
            zoom_factor = 0.5  # Adjust as needed
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * zoom_factor
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            mid_z = (z_max + z_min) / 2

            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

            ax.set_box_aspect([1, 1, 1])

            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Function to handle keyboard events
        def handle_key_event(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        fig.canvas.mpl_connect('key_press_event', handle_key_event)

        # Initial plot
        update(0)
        plt.show()

    def get_ronis_array(self, attribute, my_array, roni_frames):
        roni_CM = np.full(my_array.shape, np.nan)
        roni_CM[roni_frames] = self.roni_data[attribute][:]
        return roni_CM


def export_to_csv_and_organize(hdf5_path, output_base_dir, movies):
    with h5py.File(hdf5_path, 'r') as file:
        for movie in movies:
            movie_dir = os.path.join(output_base_dir, movie)
            os.makedirs(movie_dir, exist_ok=True)  # Create directory for the movie if it doesn't exist

            for data_attr in file[movie].keys():
                data = file[movie][data_attr][()]
                df = pd.DataFrame(data)

                if 'header' in file[movie][data_attr].attrs:
                    df.columns = file[movie][data_attr].attrs['header']

                attribute_name = data_attr.split('_')[0]  # Assuming the attribute name format is '{name}_raw'
                csv_filename = f"{movie}_{attribute_name}.csv"
                df.to_csv(os.path.join(movie_dir, csv_filename), index=False)


def fft_noise_detector(psi):
    fft = np.fft.fft(psi)
    power_spectrum = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(psi))
    high_freq_power = np.sum(power_spectrum[np.abs(freqs) > 0.1])
    low_freq_power = np.sum(power_spectrum[np.abs(freqs) <= 0.1])
    freq_ratio = high_freq_power / (low_freq_power + 1e-10)  # Avoid division by zero
    return freq_ratio


def detect_bad_signals(psi_arrays, threshold=1.5):
    bad_signals = []
    scores = {}

    for idx, key in enumerate(psi_arrays):
        psi = psi_arrays[key]
        # 1. Variance of first differences
        diff_var = np.var(np.diff(psi))

        # 2. Autocorrelation decay
        autocorr = np.correlate(psi, psi, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Positive lags only
        autocorr_decay = np.sum(np.abs(autocorr[1:] - autocorr[:-1])) / len(autocorr)

        # 3. High-frequency energy ratio
        fft = np.fft.fft(psi)
        power_spectrum = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(psi))
        high_freq_power = np.sum(power_spectrum[np.abs(freqs) > 0.1])
        low_freq_power = np.sum(power_spectrum[np.abs(freqs) <= 0.1])
        freq_ratio = high_freq_power / (low_freq_power + 1e-10)  # Avoid division by zero

        # Composite score: combine metrics (weights can be adjusted)
        score = diff_var/len(psi)  +  freq_ratio  #  + autocorr_decay
        scores[key] = score

        # Check if the signal is "bad" based on the threshold
        if score > threshold:
            bad_signals.append((idx, score))

    # Sort signals by score (descending)
    # scores.sort(key=lambda x: x[1], reverse=True)
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    return scores, bad_signals


def plot_all_roni_phi_psi():
    h5_file_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\manipulated_05_12_22.hdf5"
    output_html_base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data"

    psi_left_all = {}
    psi_right_all = {}
    phi_left_all = {}
    phi_right_all = {}
    movie_names = []

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Traverse the contents of the HDF5 file
        for movie in h5_file.keys():
            if f'{movie}/wing' in h5_file:
                # Extract the relevant columns
                phi_right_mov = h5_file[f'{movie}/wing'][:, 2]
                phi_left_mov = h5_file[f'{movie}/wing'][:, 2 + 3]
                psi_right_mov = h5_file[f'{movie}/wing'][:, 4]
                psi_left_mov = h5_file[f'{movie}/wing'][:, 4 + 3]

                # Store the data in dictionaries
                psi_left_all[movie] = psi_left_mov
                psi_right_all[movie] = psi_right_mov
                phi_left_all[movie] = phi_left_mov
                phi_right_all[movie] = phi_right_mov
                movie_names.append(movie)

    problematic = fft_noise_detector(psi=psi_left_all['mov219'])
    regular = fft_noise_detector(psi=psi_left_all['mov331'])
    a, b = detect_bad_signals(psi_left_all)
    # Define attributes to plot
    all_names = ["psi_left", "psi_right", "phi_left", "phi_right"]
    all_data = [psi_left_all, psi_right_all, phi_left_all, phi_right_all]

    # Create a plot for each attribute in groups of 10 movies
    for i in range(len(all_names)):
        attribute_name = all_names[i]
        attribute_data = all_data[i]

        # Split movie names into groups of 10
        num_groups = math.ceil(len(movie_names) / 10)
        for group_idx in range(num_groups):
            start_idx = group_idx * 10
            end_idx = min((group_idx + 1) * 10, len(movie_names))
            group_movies = movie_names[start_idx:end_idx]

            traces = []
            for movie in group_movies:
                data = attribute_data[movie]
                traces.append(go.Scatter(
                    y=data,
                    mode='lines',
                    name=movie
                ))

            layout = go.Layout(
                title=f'{attribute_name} Data (Group {group_idx + 1})',
                xaxis=dict(title='Row Index'),
                yaxis=dict(title=f'{attribute_name} Value')
            )

            fig = go.Figure(data=traces, layout=layout)

            # Save the plot to an HTML file
            output_html_path = f"{output_html_base_path}/{attribute_name}_group_{group_idx + 1}.html"
            fig.write_html(output_html_path)


def find_movies_with_high_speed():
    h5_file_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\manipulated_05_12_22.hdf5"
    speeds = {}
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        movie = 'mov206'
        psi = h5_file[f'{movie}/wing'][:, 4]
        plt.plot(psi)
        plt.show()
        # Traverse the contents of the HDF5 file
        for movie in h5_file.keys():
            if f'{movie}/body' in h5_file:
                # Extract the relevant columns
                cm_x = h5_file[f'{movie}/body'][:, 11]
                cm_y = h5_file[f'{movie}/body'][:, 12]
                cm_z = h5_file[f'{movie}/body'][:, 13]

                CM = np.column_stack((cm_x, cm_y, cm_z))
                smoothed = FlightAnalysis.savgol_smoothing(CM[:, np.newaxis, :], lam=10, polyorder=2, window_length=72, median_kernel=3, plot=False)
                smoothed = np.squeeze(smoothed)
                speed, CM_dot = FlightAnalysis.get_speed(smoothed)
                avarage = np.mean(speed)
                speeds[movie] = avarage


    speeds_sorted = dict(sorted(speeds.items(), key=lambda item: -item[1]))


def extract_and_compare_roni_psi_single(my_data_dir, roni_hdf5_path, movie_id):
    movie_key = f"{movie_id}"
    my_data_path = os.path.join(my_data_dir, movie_key, f"{movie_key}_analysis_smoothed.h5")

    # Ensure my data exists
    if not os.path.exists(my_data_path):
        print(f"My data for movie {movie_id} not found in {my_data_path}.")
        return

    # Open Roni's HDF5 file and ensure the movie exists
    with h5py.File(roni_hdf5_path, 'r') as h5_file:
        if f"{movie_key}/wing" not in h5_file or f"{movie_key}/vectors" not in h5_file:
            print(f"Roni's data for movie {movie_key} not found in {roni_hdf5_path}.")
            return

        # Extract Roni's data
        roni_group_wing = h5_file[f"{movie_key}/wing"]
        roni_group_vectors = h5_file[f"{movie_key}/vectors"]
        roni_frames = roni_group_vectors[:, 1].astype(int) - 1  # Align frames (0-based index)
        roni_psi_left = roni_group_wing[:, 4 + 3]  # psi_lw
        roni_psi_right = roni_group_wing[:, 4]  # psi_rw

    # Load my data
    with h5py.File(my_data_path, 'r') as my_hdf:
        my_psi_left = my_hdf['wings_psi_left'][:]
        my_psi_right = my_hdf['wings_psi_right'][:]

        # Ensure frame indices are within bounds
    valid_frame_indices = roni_frames[roni_frames < len(my_psi_left)]
    aligned_my_psi_left = my_psi_left[valid_frame_indices]
    aligned_my_psi_right = my_psi_right[valid_frame_indices]

    # Align lengths (in case there are mismatches after indexing)
    min_length = min(len(aligned_my_psi_left), len(roni_psi_left))
    aligned_my_psi_left = aligned_my_psi_left[:min_length]
    aligned_my_psi_right = aligned_my_psi_right[:min_length]
    roni_psi_left = roni_psi_left[:min_length]
    roni_psi_right = roni_psi_right[:min_length]

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for Left Wing Psi
    fig.add_trace(go.Scatter(
        y=roni_psi_left,
        mode='lines',
        name="Roni's Psi Left",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        y=aligned_my_psi_left,
        mode='lines',
        name="My Psi Left",
        line=dict(color='red',)
    ))

    # Add layout details
    fig.update_layout(
        title=f"Left Wing Psi Comparison for Movie {movie_id}",
        xaxis_title="Frame",
        yaxis_title="Psi (degrees)",
        legend_title="Legend",
        template="plotly_white"
    )

    # Save the figure to an HTML file
    output_html_path = f"{movie_key}_psi_comparison.html"
    fig.write_html(output_html_path)
    print(f"Plot saved to {output_html_path}")


def smoothness_score(array):
    # Calculate differences between consecutive elements
    differences = np.abs(np.diff(array))

    # Compute standard deviation of these differences
    std = np.std(differences)
    sum = np.sum(differences)

    return std, sum

def compare_psi_per_movie(my_data_path, Hull_hdf5_path, movie):
    with h5py.File(my_data_path, 'r') as my_hdf:
        my_psi_left = my_hdf['wings_psi_left'][:]
        my_psi_right = my_hdf['wings_psi_right'][:]

        my_phi_left = my_hdf['wings_phi_left'][:]
        my_phi_right = my_hdf['wings_phi_right'][:]

    with h5py.File(Hull_hdf5_path, 'r') as h5_file:
        # Extract Roni's data
        Hull_group_wing = h5_file[f"mov{movie}/wing"]
        Hull_group_vectors = h5_file[f"mov{movie}/vectors"]
        Hull_frames = Hull_group_vectors[:, 1].astype(int) - 1  # Align frames (0-based index)

        first_Hull_frame = Hull_frames[0]
        Hull_psi_left = Hull_group_wing[:, 4 + 3]  # psi_lw
        Hull_psi_right = Hull_group_wing[:, 4]

        Hull_psi_left = FlightAnalysis.add_nan_frames(Hull_psi_left, first_Hull_frame)
        Hull_psi_right = FlightAnalysis.add_nan_frames(Hull_psi_right, first_Hull_frame)

    # mutual frames
    min_length = min(len(my_phi_right), len(Hull_psi_right))
    my_phi_right, my_phi_left, my_phi_right, my_psi_left = (
        my_phi_right[:min_length],
        my_phi_left[:min_length],
        my_phi_right[:min_length],
        my_psi_left[:min_length]
    )
    Hull_psi_right, Hull_psi_left = Hull_psi_right[:min_length], Hull_psi_left[:min_length]

    # Step 1: Identify non-NaN indices for both arrays
    non_nan_indices_my_phi_right = ~np.isnan(my_phi_right + my_phi_left + my_phi_right + my_psi_left)
    non_nan_indices_Hull_psi_right = ~np.isnan(Hull_psi_right)

    # Step 2: Find common non-NaN indices
    valid_indices = np.logical_and(non_nan_indices_my_phi_right, non_nan_indices_Hull_psi_right)
    valid_indices = np.arange(len(my_phi_right))[valid_indices]

    aligned_my_psi_left = my_psi_left[valid_indices]
    aligned_my_psi_right = my_psi_right[valid_indices]
    aligned_my_phi_left = my_phi_left[valid_indices]
    aligned_my_phi_right = my_phi_right[valid_indices]

    Hull_psi_left = Hull_psi_left[valid_indices]
    Hull_psi_right = Hull_psi_right[valid_indices]

    # now the data is aligned.
    phi_s = [aligned_my_phi_left, aligned_my_phi_right]
    my_psi_s = [aligned_my_psi_left, aligned_my_psi_right]
    Hull_psi_s = [Hull_psi_left, Hull_psi_right]

    for wing in range(2):
        phi = phi_s[wing]
        my_psi = my_psi_s[wing]
        Hull_psi = Hull_psi_s[wing]
        try:
            cs_my_phi = CubicSpline(np.arange(len(phi)), phi)
            cs_my_psi = CubicSpline(np.arange(len(my_psi)), my_psi)
            cs_Hull_psi = CubicSpline(np.arange(len(Hull_psi)), Hull_psi)
        except:
            print("Spline creation error")
            continue

        min_peaks_inds, min_peak_values = FlightAnalysis.get_peaks(-phi, 0, len(phi) - 1, show=False, prominence=75)
        for wing_bit in range(len(min_peaks_inds) - 1):
            start = min_peaks_inds[wing_bit]
            end = min_peaks_inds[wing_bit + 1]
            time_stamps = np.linspace(start=start, stop=end, num=100)
            my_psi_wingbit = cs_my_psi(time_stamps)
            Hull_psi_wingbit = cs_Hull_psi(time_stamps)
            my_phi_wingbit = cs_my_phi(time_stamps)

            all_psi_wingbits_my.append(my_psi_wingbit)
            all_psi_wingbits_Hull.append(Hull_psi_wingbit)
            all_phi_wingbit.append(my_phi_wingbit)

def compare_psi_smoothness(specific_data_path=None):
    
    Hull_hdf5_path = fr"/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/comparison_data/manipulated_05_12_22.hdf5"
    all_psi_wingbits_my = []
    all_psi_wingbits_Hull = []
    all_phi_wingbit = []
    my_data_dir = fr"/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/predict_output/debug_outputs/movie_1_10_4898_ds_3tc_7tj/movie_1_10_4898_ds_3tc_7tj"
    movies = [1, 2, 8, 10, 23, 122, 165, 166, 200, 202, 246, 264, 494, 522, 529, 531]
        
    for movie in movies:
        print(f"movie is {movie}")
        my_data_path = os.path.join(my_data_dir, f"mov{movie}",  f"mov{movie}_analysis_smoothed.h5")
        with h5py.File(my_data_path, 'r') as my_hdf:
            my_psi_left = my_hdf['wings_psi_left'][:]
            my_psi_right = my_hdf['wings_psi_right'][:]

            my_phi_left = my_hdf['wings_phi_left'][:]
            my_phi_right = my_hdf['wings_phi_right'][:]

        with h5py.File(Hull_hdf5_path, 'r') as h5_file:
            # Extract Roni's data
            Hull_group_wing = h5_file[f"mov{movie}/wing"]
            Hull_group_vectors = h5_file[f"mov{movie}/vectors"]
            Hull_frames = Hull_group_vectors[:, 1].astype(int) - 1  # Align frames (0-based index)

            first_Hull_frame = Hull_frames[0]
            Hull_psi_left = Hull_group_wing[:, 4 + 3]  # psi_lw
            Hull_psi_right = Hull_group_wing[:, 4]

            Hull_psi_left = FlightAnalysis.add_nan_frames(Hull_psi_left, first_Hull_frame)
            Hull_psi_right = FlightAnalysis.add_nan_frames(Hull_psi_right, first_Hull_frame)

        # mutual frames
        min_length = min(len(my_phi_right), len(Hull_psi_right))
        my_phi_right, my_phi_left, my_phi_right, my_psi_left = (
            my_phi_right[:min_length],
            my_phi_left[:min_length],
            my_phi_right[:min_length],
            my_psi_left[:min_length]
        )
        Hull_psi_right, Hull_psi_left = Hull_psi_right[:min_length], Hull_psi_left[:min_length]

        # Step 1: Identify non-NaN indices for both arrays
        non_nan_indices_my_phi_right = ~np.isnan(my_phi_right + my_phi_left + my_phi_right + my_psi_left)
        non_nan_indices_Hull_psi_right = ~np.isnan(Hull_psi_right)

        # Step 2: Find common non-NaN indices
        valid_indices = np.logical_and(non_nan_indices_my_phi_right, non_nan_indices_Hull_psi_right)
        valid_indices = np.arange(len(my_phi_right))[valid_indices]

        aligned_my_psi_left = my_psi_left[valid_indices]
        aligned_my_psi_right = my_psi_right[valid_indices]
        aligned_my_phi_left = my_phi_left[valid_indices]
        aligned_my_phi_right = my_phi_right[valid_indices]

        Hull_psi_left = Hull_psi_left[valid_indices]
        Hull_psi_right = Hull_psi_right[valid_indices]

        # now the data is aligned.
        phi_s = [aligned_my_phi_left, aligned_my_phi_right]
        my_psi_s = [aligned_my_psi_left, aligned_my_psi_right]
        Hull_psi_s = [Hull_psi_left, Hull_psi_right]

        for wing in range(2):
            phi = phi_s[wing]
            my_psi = my_psi_s[wing]
            Hull_psi = Hull_psi_s[wing]
            try:
                cs_my_phi = CubicSpline(np.arange(len(phi)), phi)
                cs_my_psi = CubicSpline(np.arange(len(my_psi)), my_psi)
                cs_Hull_psi = CubicSpline(np.arange(len(Hull_psi)), Hull_psi)
            except:
                print("Spline creation error")
                continue

            min_peaks_inds, min_peak_values = FlightAnalysis.get_peaks(-phi, 0, len(phi) - 1, show=False, prominence=75)
            for wing_bit in range(len(min_peaks_inds) - 1):
                start = min_peaks_inds[wing_bit]
                end = min_peaks_inds[wing_bit + 1]
                time_stamps = np.linspace(start=start, stop=end, num=100)
                my_psi_wingbit = cs_my_psi(time_stamps)
                Hull_psi_wingbit = cs_Hull_psi(time_stamps)
                my_phi_wingbit = cs_my_phi(time_stamps)

                all_psi_wingbits_my.append(my_psi_wingbit)
                all_psi_wingbits_Hull.append(Hull_psi_wingbit)
                all_phi_wingbit.append(my_phi_wingbit)

    all_psi_wingbits_my = np.array(all_psi_wingbits_my).T
    all_psi_wingbits_Hull = np.array(all_psi_wingbits_Hull).T
    all_phi_wingbit = np.array(all_phi_wingbit).T

    # -------- FIRST FIGURE -----------------------------------------
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)  # Increased DPI for better resolution

    # Define colors for distinction
    color_my = 'blue'
    color_Hull = 'red'
    color_phi = 'green'

    point_size = 2
    alpha_value = 0.6
    scatter = True
    plot_Std = True

    # Arrays to store means and stds for plotting
    x_range = np.arange(100)
    means_my = np.zeros(100)
    means_Hull = np.zeros(100)
    means_phi = np.zeros(100)
    stds_my = np.zeros(100)
    stds_Hull = np.zeros(100)
    stds_phi = np.zeros(100)

    # Calculate means and stds for each time step
    for i in range(100):
        y_values_phi = all_phi_wingbit[i]
        y_values_my = all_psi_wingbits_my[i]
        y_values_Hull = all_psi_wingbits_Hull[i]

        means_my[i] = np.mean(y_values_my)
        means_Hull[i] = np.mean(y_values_Hull)
        means_phi[i] = np.mean(y_values_phi)

        stds_my[i] = np.std(y_values_my)
        stds_Hull[i] = np.std(y_values_Hull)
        stds_phi[i] = np.std(y_values_phi)

        # Original scatter plots if enabled
        if scatter:
            x_values_phi = np.full(all_phi_wingbit.shape[1], i - 0.3)
            x_values_my = np.full(all_psi_wingbits_my.shape[1], i)
            x_values_Hull = np.full(all_psi_wingbits_Hull.shape[1], i + 0.3)

            ax.scatter(x_values_my, y_values_my, color=color_my, alpha=alpha_value,
                       label='Our Psi Data' if i == 0 else "", s=point_size)
            ax.scatter(x_values_Hull, y_values_Hull, color=color_Hull, alpha=alpha_value,
                       label='Hull\'s Psi Data' if i == 0 else "", s=point_size)

    # Plot means as lines
    ax.plot(x_range, means_my, color=color_my, label='Our Psi Mean', linewidth=2)
    ax.plot(x_range, means_Hull, color=color_Hull, label='Hull Psi Mean', linewidth=2)
    ax.plot(x_range, means_phi, color=color_phi, label='Phi Mean', linewidth=2)

    # Plot standard deviation ranges if enabled
    if plot_Std:
        ax.fill_between(x_range, means_my - stds_my, means_my + stds_my,
                        color=color_my, alpha=0.2)
        ax.fill_between(x_range, means_Hull - stds_Hull, means_Hull + stds_Hull,
                        color=color_Hull, alpha=0.2)

    # Calculate average STDs
    avg_std_my = np.mean(stds_my)
    avg_std_Hull = np.mean(stds_Hull)

    # Create base title and filename
    base_title = "Hull vs Our method's psi distribution during a wing-bit"
    full_title = f"{base_title}\nAvg STD - Our's: {avg_std_my:.3f} [deg], Hull: {avg_std_Hull:.3f} [deg]"

    # Slightly reduced font sizes for titles, labels, and ticks
    ax.set_title(full_title, pad=20, fontsize=18)
    ax.set_xlabel('Time Step [percentage of a wing beat]', fontsize=14)
    ax.set_ylabel('degrees', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)  # Tick label size

    ax.legend(loc='upper right', fontsize=12)  # Legend font size
    ax.set_xlim([-1, 100])
    plt.tight_layout()  # Adjust layout to make room for label rotation

    # Save first figure
    plt.savefig(f"{base_title}.png",
                dpi=600,  # High DPI for print-quality
                bbox_inches='tight',
                format='png')
    plt.close()

    #################################################################
    # Second figure: "Distribution of deviations from the mean ψ"
    #################################################################
    fig2, ax2 = plt.subplots(figsize=(15, 8), dpi=600)

    # Plot the data around zero
    for i in range(100):
        # Center around zero
        y_values_my = all_psi_wingbits_my[i] - means_my[i]
        y_values_hull = all_psi_wingbits_Hull[i] - means_Hull[i]
        x_values_my = np.full_like(y_values_my, i + 0.3, dtype=float)
        x_values_hull = np.full_like(y_values_hull, i - 0.3, dtype=float)

        if i == 0:
            ax2.scatter(x_values_my, y_values_my, color='blue', alpha=alpha_value,
                        s=point_size, label='Our Psi Data')
            ax2.scatter(x_values_hull, y_values_hull, color='red', alpha=alpha_value,
                        s=point_size, label="Hull's Psi Data")
        else:
            ax2.scatter(x_values_my, y_values_my, color='blue', alpha=alpha_value, s=point_size)
            ax2.scatter(x_values_hull, y_values_hull, color='red', alpha=alpha_value, s=point_size)

    # ± std fill, centered at 0
    ax2.fill_between(
        x_range,
        stds_my, -stds_my,
        color='blue', alpha=0.2, label='Our ψ Std'
    )
    ax2.fill_between(
        x_range,
        stds_Hull, -stds_Hull,
        color='red', alpha=0.2, label="Hull's ψ Std"
    )

    # Zero line
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Zero Line')

    base_title_2 = "Distribution of deviations of ψ measurements from the mean ψ, along a wing-bit"
    full_title_2 = (f"{base_title_2}\n"
                    f"Avg STD - Our: {avg_std_my:.3f} [deg], Hull: {avg_std_Hull:.3f} [deg]")

    # Slightly reduced font sizes here as well
    ax2.set_title(full_title_2, pad=20, fontsize=18)
    ax2.set_xlabel("Time Step [percentage of a wing beat]", fontsize=14)
    ax2.set_ylabel("degrees", fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    ax2.set_xticks(range(0, 100, 5))
    ax2.set_xticklabels(range(0, 100, 5), rotation=90)
    ax2.set_xlim([-1, 100])

    plt.tight_layout()
    second_filename = "Distribution_of_deviations_of_ψ_measurements.png"
    print(f"Saving high-res PNG to: {second_filename}")
    plt.savefig(second_filename, format='png', dpi=600, bbox_inches='tight')
    plt.close()

    #############################################################################################################
    # save in plotly
    #############################################################################################################
    # Create the figure
    # fig = go.Figure()
    #
    # if scatter:
    #     for i in range(0, 100):
    #         y_values_my = all_psi_wingbits_my[i]
    #         y_values_Hull = all_psi_wingbits_Hull[i]
    #
    #         # My data points
    #         fig.add_trace(go.Scatter(
    #             x=[i + 0.3] * len(y_values_my),
    #             y=y_values_my,
    #             mode='markers',
    #             name='Our Data Points',
    #             marker=dict(color='blue', size=point_size),
    #             opacity=alpha_value,
    #             legendgroup='my_data',
    #             showlegend=(i == 0)
    #         ))
    #
    #         # Hull's data points
    #         fig.add_trace(go.Scatter(
    #             x=[i - 0.3] * len(y_values_Hull),
    #             y=y_values_Hull,
    #             mode='markers',
    #             name="Hull's Data Points",
    #             marker=dict(color='red', size=point_size),
    #             opacity=alpha_value,
    #             legendgroup='Hull_data',
    #             showlegend=(i == 0)
    #         ))
    #
    # # Add mean lines (still show all points for the continuous lines)
    # fig.add_trace(go.Scatter(
    #     x=x_range,
    #     y=means_my,
    #     mode='lines',
    #     name='Our ψ Mean',
    #     line=dict(color='blue', width=2)
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=x_range,
    #     y=means_Hull,
    #     mode='lines',
    #     name='Hull ψ Mean',
    #     line=dict(color='red', width=2)
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=x_range,
    #     y=means_phi,
    #     mode='lines',
    #     name='φ Mean',
    #     line=dict(color='green', width=2)
    # ))
    #
    # # Add standard deviation ranges if enabled
    # if plot_Std:
    #     # My data std
    #     fig.add_trace(go.Scatter(
    #         x=np.concatenate([x_range, x_range[::-1]]),
    #         y=np.concatenate([means_my + stds_my, (means_my - stds_my)[::-1]]),
    #         fill='toself',
    #         fillcolor='rgba(0,0,255,0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         name='Our ψ Std',
    #         showlegend=True
    #     ))
    #
    #     # Roni's data std
    #     fig.add_trace(go.Scatter(
    #         x=np.concatenate([x_range, x_range[::-1]]),
    #         y=np.concatenate([means_Hull + stds_Hull, (means_Hull - stds_Hull)[::-1]]),
    #         fill='toself',
    #         fillcolor='rgba(255,0,0,0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         name='Hull ψ Std',
    #         showlegend=True
    #     ))
    #
    #     # Phi data std
    #     # fig.add_trace(go.Scatter(
    #     #     x=np.concatenate([x_range, x_range[::-1]]),
    #     #     y=np.concatenate([means_phi + stds_phi, (means_phi - stds_phi)[::-1]]),
    #     #     fill='toself',
    #     #     fillcolor='rgba(0,255,0,0.2)',
    #     #     line=dict(color='rgba(255,255,255,0)'),
    #     #     name='φ Std',
    #     #     showlegend=True
    #     # ))
    #
    # # Update layout
    # base_title = "Hull vs Our ψ distribution during a wing-bit"
    # full_title = f"{base_title}<br>Avg STD - Our: {avg_std_my:.3f}, Hull: {avg_std_Hull:.3f}"
    #
    # fig.update_layout(
    #     title=dict(
    #         text=full_title,
    #         y=0.95,
    #         x=0.5,
    #         xanchor='center',
    #         yanchor='top'
    #     ),
    #     xaxis_title='Time Step',
    #     yaxis_title='ψ Values',
    #     showlegend=True,
    #     legend=dict(
    #         yanchor="top",
    #         y=0.99,
    #         xanchor="right",
    #         x=0.99
    #     ),
    #     width=1200,
    #     height=800,
    #     xaxis=dict(
    #         showgrid=False,
    #         zeroline=False,
    #         range=[0, max(x_range)],
    #     ),
    #     yaxis=dict(
    #         showgrid=False,
    #         zeroline=False
    #     ),
    #     plot_bgcolor='white',
    #     paper_bgcolor='white'
    # )
    #
    # # Update x-axis
    # fig.update_xaxes(
    #     tickmode='linear',
    #     tick0=0,
    #     dtick=5
    # )
    #
    # # Save as HTML
    # print(f"write to {base_title}.html")
    # fig.write_html(f"{base_title}.html")
    # print(f"write to {base_title}.pdf")
    # fig.write_image(f"{base_title}.pdf", format="pdf", width=1200, height=800, scale=2)

    # # Optionally, also save as PNG for compatibility
    # fig.write_image(f"{base_title}.png", scale=3)  # scale=3 for high resolution

    # Create second figure for centered data
    # fig2 = go.Figure()
    #
    # # Add scatter points if enabled, centered around 0 (subtract mean from each point)
    # if scatter:
    #     for i in range(0, 100):
    #         y_values_my = all_psi_wingbits_my[i] - means_my[i]  # Center around 0
    #         y_values_Hull = all_psi_wingbits_Hull[i] - means_Hull[i]  # Center around 0
    #
    #         # My data points
    #         fig2.add_trace(go.Scatter(
    #             x=[i + 0.3] * len(y_values_my),
    #             y=y_values_my,
    #             mode='markers',
    #             name='Our Data Points',
    #             marker=dict(color='blue', size=point_size),
    #             opacity=alpha_value,
    #             legendgroup='my_data',
    #             showlegend=(i == 0)
    #         ))
    #
    #         # Hull's data points
    #         fig2.add_trace(go.Scatter(
    #             x=[i - 0.3] * len(y_values_Hull),
    #             y=y_values_Hull,
    #             mode='markers',
    #             name="Hull's Data Points",
    #             marker=dict(color='red', size=point_size),
    #             opacity=alpha_value,
    #             legendgroup='Hull_data',
    #             showlegend=(i == 0)
    #         ))
    #
    #     # Add standard deviation ranges
    #     # My data std (centered around 0)
    #     fig2.add_trace(go.Scatter(
    #         x=np.concatenate([x_range, x_range[::-1]]),
    #         y=np.concatenate([stds_my, -stds_my[::-1]]),
    #         fill='toself',
    #         fillcolor='rgba(0,0,255,0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         name='My ψ Std',
    #         showlegend=True
    #     ))
    #
    #     # Hull's data std (centered around 0)
    #     fig2.add_trace(go.Scatter(
    #         x=np.concatenate([x_range, x_range[::-1]]),
    #         y=np.concatenate([stds_Hull, -stds_Hull[::-1]]),
    #         fill='toself',
    #         fillcolor='rgba(255,0,0,0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         name='Hull ψ Std',
    #         showlegend=True
    #     ))
    #
    #     # Add zero line for reference
    #     fig2.add_trace(go.Scatter(
    #         x=x_range,
    #         y=np.zeros_like(x_range),
    #         mode='lines',
    #         name='Zero Line',
    #         line=dict(color='black', width=1, dash='dash'),
    #         showlegend=True
    #     ))
    #
    #     # Update layout for the second figure
    #     base_title_2 = "Distribution of deviations of ψ measurements from the mean ψ, along a wing-bit"
    #     full_title_2 = f"{base_title_2}<br>Avg STD - Our: {avg_std_my:.3f}, Hull: {avg_std_Hull:.3f}"
    #
    #     fig2.update_layout(
    #         title=dict(
    #             text=base_title_2,
    #             y=0.95,
    #             x=0.5,
    #             xanchor='center',
    #             yanchor='top'
    #         ),
    #         xaxis_title='Time Step',
    #         yaxis_title='Deviation from Mean',
    #         showlegend=True,
    #         legend=dict(
    #             yanchor="top",
    #             y=0.99,
    #             xanchor="right",
    #             x=0.99
    #         ),
    #         width=1200,
    #         height=800,
    #         xaxis=dict(
    #             showgrid=False,
    #             zeroline=False,
    #             range=[0, max(x_range)],
    #         ),
    #         yaxis=dict(
    #             showgrid=False,
    #             zeroline=False
    #         ),
    #         plot_bgcolor='white',
    #         paper_bgcolor='white'
    #     )
    #
    #     # Update x-axis
    #     fig2.update_xaxes(
    #         tickmode='linear',
    #         tick0=0,
    #         dtick=5
    #     )
    #
    #     # Save as HTML
    #     print(f"write to {base_title_2}.html")
    #     fig2.write_html(f"{base_title_2}.html")
    #     print(f"write to {base_title_2}.png")
    #     fig2.write_image(
    #         f"{base_title_2}.png",
    #         format="png",
    #         width=1200,  # base width
    #         height=800,  # base height
    #         scale=1
    #     )
        # print(f"write to {base_title_2}.pdf")
        # fig2.write_image(f"{base_title_2}.pdf", format="pdf", width=1200, height=800, scale=2)


if __name__ == "__main__":
    # find_movies_with_high_speed()
    compare_psi_smoothness()
    my_data_dir = fr"/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/predict_output/debug_outputs/movie_1_10_4898_ds_3tc_7tj/movie_1_10_4898_ds_3tc_7tj"
    hull_hdf5_path = fr"/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/comparison_data/manipulated_05_12_22.hdf5"
    movie_id = "movie_1_10_4898_ds_3tc_7tj"
    extract_and_compare_roni_psi_single(my_data_dir, hull_hdf5_path, movie_id)

    hdf5_path = fr"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\Roni analisys\cliped_2023_08_09_60ms.hdf5"
    output_base_dir = fr"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\Roni analisys\not smoothed"
    movies = ["mov78", "mov101", "mov104"]
    export_to_csv_and_organize(hdf5_path, output_base_dir, movies)
    plot_all_roni_phi_psi()
    path_my_data = fr"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys"
    path_roni_data = fr"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\Roni analisys"
    movies = [78, 101, 104]
    mov_num = movies[1]
    smoothed = True
    
    comparison = DataComparison(mov_num, path_my_data, path_roni_data, smoothed, smooth_like_roni=smoothed)
    comparison.plot_body_pitch_phi_right_phi_left()
    comparison.plot_body_roll_phi_right_phi_left()
    comparison.plot_body_pitch_theta_left_theta_right()
    #
    comparison.visualize_3D_comparison()
    comparison.compare_CM()
    comparison.compare_CM_velocity()
    comparison.compare_body_vectors()
    comparison.compare_body_angles()
    comparison.compare_stroke_planes()
    comparison.compare_wings_span(wing='left')
    comparison.theta_vs_phi()
    comparison.compare_wings_angles(wing='right')
    comparison.compare_wings_angles(wing='left')
    comparison.compare_wings_chords(wing='right')
    comparison.compare_wings_tip(wing='right')
