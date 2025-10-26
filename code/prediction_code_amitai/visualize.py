import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib import colors
from matplotlib.widgets import Slider, Button
from skimage import morphology
# from scipy.spatial import ConvexHull
from utils import predict_3D_points_all_pairs
import h5py
import plotly.graph_objects as go
import plotly.io as pio
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R
import os
import cv2
import itertools
from skimage import measure
from plotly.subplots import make_subplots
from scipy.integrate import simpson
from utils import add_nan_frames
import glob
from scipy.optimize import curve_fit
from scipy.linalg import svd
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from moviepy import VideoFileClip
import re
np.random.seed(42)
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
               (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
               (16, 17)]
SAVE = "SAVE"
DISPLAY = "DISPLAY"
POINTS_2_PLANES = [[0, 1, 2, 3, 4, 5, 6],
                   [8, 9, 10, 11, 12, 13, 14]]

POINTS_4_PLANES = [[0, 1, 2, 6], [2, 3, 4, 5, 6],
                   [8, 9, 10, 14], [10, 11, 12, 13, 14]]

INPUT = 1
OUTPUT = 0
#
# POINTS_4_PLANES = [[0, 1, 2, 3], [3, 4, 5, 6],
#                    [8, 9, 10, 11], [11, 12, 13, 14]]


class Visualizer:
    @staticmethod
    def show_predictions_one_cam(box, points_2D):
        # Assuming movie is your 5D numpy array and points is your 3D array
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        camera = [0]  # initial camera as a one-element list

        def update(val):
            frame = slider.val
            ax.clear()  # clear the previous scatter points
            ax.imshow(movie[int(slider.val), camera[0]])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(points[int(slider.val), camera[0]])))  # create a color array
            ax.scatter(*points[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='o')  # scatter points on image
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value
            elif event.key == 'up':
                camera[0] = min(camera[0] + 1, movie.shape[1] - 1)  # switch to next camera
                update(None)
            elif event.key == 'down':
                camera[0] = max(camera[0] - 1, 0)  # switch to previous camera
                update(None)

        fig, ax = plt.subplots(figsize=(10, 10))  # single camera view
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_all_cams(box, points_2D, scatter=True):
        # movie = Visualizer.get_display_box(box)
        movie = box[..., [1, 1, 1]]
        points = points_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(1-movie[frame, i])
                if scatter:
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(points[frame, i])))  # create a color array
                    ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='.')  #
                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_vs_reprojections_one_cam(box, points_2D, reprojections_2D):
        # Assuming movie is your 5D numpy array and points is your 3D array
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]
        reprojections = reprojections_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        camera = [0]  # initial camera as a one-element list

        def update(val):
            frame = slider.val
            ax.clear()  # clear the previous scatter points
            ax.imshow(movie[int(slider.val), camera[0]])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(points[int(slider.val), camera[0]])))  # create a color array
            ax.scatter(*points[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='o')  # scatter points on image
            ax.scatter(*reprojections[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='d')  # scatter points on image
            for point, point_reprojected in zip(points[frame, camera[0]], reprojections[frame, camera[0]]):
                ax.plot(*zip(point, point_reprojected), color='yellow')
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value
            elif event.key == 'up':
                camera[0] = min(camera[0] + 1, movie.shape[1] - 1)  # switch to next camera
                update(None)
            elif event.key == 'down':
                camera[0] = max(camera[0] - 1, 0)  # switch to previous camera
                update(None)

        fig, ax = plt.subplots(figsize=(10, 10))  # single camera view
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_vs_reprejections_all_cams(box, points_2D, points_2D_reprojected):
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]
        points_reprojected = points_2D_reprojected[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
                colors = plt.cm.rainbow(np.linspace(0, 1, len(points[frame, :])))  # create a color array for points_2D
                ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='o')  # display points_2D
                ax.scatter(*points_reprojected[frame, i].T, edgecolors=colors, facecolors='none', marker='d')
                # Add a line between every corresponding point in points and points_reprojected
                for point, point_reprojected in zip(points[frame, i], points_reprojected[frame, i]):
                    ax.plot(*zip(point, point_reprojected), color='yellow')

            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_points_in_3D(points):
        # Assuming points is your (N, M, 3) array
        # Calculate the limits of the plot
        x_min, y_min, z_min = np.nanmin(points, axis=(0, 1))
        x_max, y_max, z_max = np.nanmax(points, axis=(0, 1))

        # Create a color array
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        # connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
        #                (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 7),
        #                (14, 15)]
        connections = [(0,1), (1,2), (2,3),   (3,4),   (4,5),   (5,6),  (0,6),
                        (8,9), (9,10),(10,11), (11,12), (12,13), (13,14), (8,14),
                        (7, 15),
                        (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            points_to_plot = points[[frame], :, :]
            for i in range(num_points):
                ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')
            # Reset the limits of the plot
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for 3D plot
            ax.set_aspect('equal')
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

    @staticmethod
    def create_closed_spline(points, num_points=100):
        points = np.vstack([points, points[0]])
        t = np.linspace(0, 1, len(points))
        spline = make_interp_spline(t, points, bc_type='periodic')
        t_new = np.linspace(0, 1, num_points)
        spline_points = spline(t_new)
        return spline_points

    @staticmethod
    def create_ellipsoid(center, radii, rotation_matrix, num_points=100):
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)
        x = radii[0] * np.outer(np.sin(v), np.cos(u))
        y = radii[1] * np.outer(np.sin(v), np.sin(u))
        z = radii[2] * np.outer(np.cos(v), np.ones_like(u))

        # Rotate and translate points
        ellipsoid = np.dot(rotation_matrix, np.vstack([x.flatten(), y.flatten(), z.flatten()]))
        x, y, z = ellipsoid.reshape((3, num_points, num_points)) + center[:, np.newaxis, np.newaxis]
        return x, y, z

    @staticmethod
    def create_sphere(center, radius, num_points=100):
        phi = np.linspace(0, 2 * np.pi, num_points)
        theta = np.linspace(0, np.pi, num_points)
        x = center[0] + radius * np.outer(np.sin(theta), np.cos(phi))
        y = center[1] + radius * np.outer(np.sin(theta), np.sin(phi))
        z = center[2] + radius * np.outer(np.cos(theta), np.ones_like(phi))
        return x, y, z

    @staticmethod
    def show_points_in_3D_special(points, plot_points=False):
        x_min, y_min, z_min = points.min(axis=(0, 1))
        x_max, y_max, z_max = points.max(axis=(0, 1))

        color_array = colors.hsv_to_rgb(
            np.column_stack((np.linspace(0, 1, points.shape[1]), np.ones((points.shape[1], 2)))))

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       (7, 15), (16, 17)]

        slider = Slider(plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow'), 'Frame', 0,
                        len(points) - 1, valinit=0, valstep=1)

        def update(val):
            ax.cla()
            frame = int(slider.val)
            points_to_plot = points[frame, :, :]
            if plot_points:
                for i in range(len(points_to_plot)):
                    ax.scatter(*points_to_plot[i], color=color_array[i])
                for i, j in connections:
                    ax.plot(*points[frame, [i, j], :].T, color='k')

            # Fit and plot splines for each wing
            wing1_indices = range(0, 6)
            wing2_indices = range(8, 14)
            wing1_points = points[frame, wing1_indices, :]
            wing2_points = points[frame, wing2_indices, :]

            wing1_spline = Visualizer.create_closed_spline(wing1_points)
            wing2_spline = Visualizer.create_closed_spline(wing2_points)

            ax.plot(wing1_spline[:, 0], wing1_spline[:, 1], wing1_spline[:, 2], 'r-', label='Wing 1 Spline',
                    linewidth=3)
            ax.plot(wing2_spline[:, 0], wing2_spline[:, 1], wing2_spline[:, 2], 'g-', label='Wing 2 Spline',
                    linewidth=3)

            # Body ellipsoid
            body_center = (points_to_plot[16] + points_to_plot[17]) / 2
            body_length = np.linalg.norm(points_to_plot[17] - points_to_plot[16])
            body_radii = [body_length / 2, body_length / 6, body_length / 6]
            direction = (points_to_plot[17] - points_to_plot[16])
            body_rotation = R.align_vectors([direction], [[1, 0, 0]])[0].as_matrix()
            x, y, z = Visualizer.create_ellipsoid(body_center, body_radii, body_rotation)
            ax.plot_surface(x, y, z, color='blue', alpha=0.5)

            # Head sphere
            head_radius = body_length / 5
            x, y, z = Visualizer.create_sphere(points_to_plot[17], head_radius)
            ax.plot_surface(x, y, z, color='red', alpha=0.5)

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_box_aspect([1, 1, 1])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event', lambda event: slider.set_val(
            min(slider.val + 1, slider.valmax)) if event.key == 'right' else slider.set_val(
            max(slider.val - 1, slider.valmin)))

        update(0)
        plt.show()

    @staticmethod
    def show_points_in_3D_projections(points):
        # Assuming points is your (N, M, 3) array
        # Calculate the limits of the plot
        x_min, y_min, z_min = points.min(axis=(0, 1))
        x_max, y_max, z_max = points.max(axis=(0, 1))

        # Create a color array
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
                       (7, 15),
                       (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        # Create the play button
        axplay = plt.axes([0.85, 0.95, 0.1, 0.03])
        button = Button(axplay, 'Play', color='lightgoldenrodyellow', hovercolor='0.975')

        # Create the speed control
        axspeed = plt.axes([0.05, 0.95, 0.1, 0.03], facecolor='lightgoldenrodyellow')
        speed = Slider(axspeed, 'Speed', 0.5, 100, valinit=10, valstep=0.5)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            points_to_plot = points[frame, :, :]
            for i in range(num_points):
                ax.scatter(points_to_plot[i, 0], points_to_plot[i, 1], points_to_plot[i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points_to_plot[[i, j], 0], points_to_plot[[i, j], 1], points_to_plot[[i, j], 2], c='k')
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            fig.canvas.draw_idle()

            # Plot the projections on the walls
            # XY plane
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], z_min, c=color_array, alpha=0.2)
            # XZ plane
            ax.scatter(points_to_plot[:, 0], y_min, points_to_plot[:, 2], c=color_array, alpha=0.2)
            # YZ plane
            ax.scatter(x_min, points_to_plot[:, 1], points_to_plot[:, 2], c=color_array, alpha=0.2)

            # Plot the connections on the projections
            # XY plane
            for i, j in connections:
                ax.plot(points_to_plot[[i, j], 0], points_to_plot[[i, j], 1], [z_min, z_min], c='b', alpha=0.5)
            # XZ plane
            for i, j in connections:
                ax.plot(points_to_plot[[i, j], 0], [y_min, y_min], points_to_plot[[i, j], 2], c='b', alpha=0.5)
            # YZ plane
            for i, j in connections:
                ax.plot([x_min, x_min], points_to_plot[[i, j], 1], points_to_plot[[i, j], 2], c='b', alpha=0.5)
            fig.canvas.draw_idle()

        def play(event):
            # Play the animation by updating the slider value
            nonlocal slider
            for i in range(slider.valmin, slider.valmax + 1):
                slider.set_val(i)
                # Adjust the speed according to the slider value
                plt.pause(1 / speed.val)

        # Connect the update functions to the widgets
        slider.on_changed(update)
        button.on_clicked(play)
        speed.on_changed(update)

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

    @staticmethod
    def show_points_and_wing_planes_3D(points, planes):
        # planes is an array (num_frames, num_planes, 4)
        # Assuming points is your (num_frames, num_joints, 3)
        # Calculate the limits of the plot

        num_planes = planes.shape[1]

        # POINTS = POINTS_2_PLANES if num_planes == 2 else if  POINTS_4_PLANES
        if num_planes == 2:
            POINTS = POINTS_2_PLANES
        elif num_planes == 4:
            POINTS = POINTS_4_PLANES
        else:
            POINTS = [np.arange(18)]

        x_min, y_min, z_min = np.nanmin(points, axis=(0, 1))
        x_max, y_max, z_max = np.nanmax(points, axis=(0, 1))

        # Create a color array
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        connections = [(0, 1), (1, 2), (2, 3),   (3,4),   (4,5),   (5, 6),  (0, 6),
                        (8, 9), (9, 10), (10, 11), (11,12), (12,13), (13, 14), (8, 14),
                        (7, 15),
                        (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            for i in range(num_points):
                ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')
            # Reset the limits of the plot

            # add plane
            for plane_num in range(num_planes):
                wing_points = points[frame, POINTS[plane_num], :]
                min_x, min_y, min_z = np.amin(wing_points, axis=0)
                max_x, max_y, max_z = np.amax(wing_points, axis=0)
                a, b, c, d = planes[frame, plane_num, :]
                xx, yy = np.meshgrid(np.linspace(min_x, max_x, 50), np.linspace(min_y, max_y, 50))
                zz = (-a*xx - b*yy - d)/c
                zz[zz > max_z] = np.nan
                zz[zz < min_z] = np.nan
                ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, label='Plane')

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
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

    @staticmethod
    def get_display_box(box):
        masks = box[..., -2:]
        num_frames, num_cams, _, _, num_masks = masks.shape
        for frame in range(num_frames):
            for cam in range(num_cams):
                for wing in range(num_masks):
                    mask = masks[frame, cam, :, :, wing]
                    dilated = morphology.binary_dilation(mask)
                    eroded = morphology.binary_erosion(mask)
                    perimeters = dilated ^ eroded
                    masks[frame, cam, :, :, wing] = perimeters
        box[..., -2:] = masks
        box[..., -2:] += box[..., [1, 1]]
        movie = box[..., [1, 3, 4]]
        movie[movie > 1] = 1
        return movie

    @staticmethod
    def display_movie_from_box(box):
        movie = box
        # movie = box[..., [1, 1, 1]]
        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def display_movie_from_path(path):
        movie = Visualizer.get_box(path)

        # movie = box[..., [1, 1, 1]]
        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def get_box(path, frames=None):
        if frames is None:
            box = h5py.File(path, "r")["/box"][:]
        else:
            box = h5py.File(path, "r")["/box"][frames]
        box = np.transpose(box, (0, 3, 2, 1))
        x1 = np.expand_dims(box[:, :, :, 0:3], axis=1)
        x2 = np.expand_dims(box[:, :, :, 3:6], axis=1)
        x3 = np.expand_dims(box[:, :, :, 6:9], axis=1)
        x4 = np.expand_dims(box[:, :, :, 9:12], axis=1)
        box = np.concatenate((x1, x2, x3, x4), axis=1)
        return box

    @staticmethod
    def create_movie_plot(com, x_body, y_body, points_3D, start_frame, save_path):
        facing = -np.cross(x_body, y_body, axis=1)
        left_tip = points_3D[:, 3, :]
        right_tip = points_3D[:, 3 + 8, :]
        num_frames = len(points_3D)
        frame0 = min(340 - start_frame, num_frames - 1)
        frame_40_ms = min(frame0 + 40 * 16, num_frames - 1)
        interval = 70
        # Create traces
        marker_size = 2
        mode = 'lines'
        trace_com = go.Scatter3d(x=com[:, 0], y=com[:, 1], z=com[:, 2], mode='markers', name='Center of Mass',
                                 marker=dict(size=marker_size), line=dict(width=5))
        trace_left_tip = go.Scatter3d(x=left_tip[:, 0], y=left_tip[:, 1], z=left_tip[:, 2], mode=mode,
                                      name='left tip', marker=dict(size=marker_size), line=dict(width=5))
        trace_right_tip = go.Scatter3d(x=right_tip[:, 0], y=right_tip[:, 1], z=right_tip[:, 2], mode=mode,
                                       name='right tip', marker=dict(size=marker_size), line=dict(width=5))
        # Create markers
        marker_start = go.Scatter3d(x=[com[0, 0]], y=[com[0, 1]], z=[com[0, 2]], mode='markers',
                                    marker=dict(size=8, color='red'), name='start')
        marker_start_dark = go.Scatter3d(x=[com[frame0, 0]], y=[com[frame0, 1]], z=[com[frame0, 2]], mode='markers',
                                         marker=dict(size=8, color='orange'), name='start dark')
        marker_40_ms = go.Scatter3d(x=[com[frame_40_ms, 0]], y=[com[frame_40_ms, 1]], z=[com[frame_40_ms, 2]],
                                    mode='markers', marker=dict(size=8, color='yellow'), name='+ 40 ms')
        # Create quiver plot
        name = 'body yaw pitch'
        size = 0.003

        quiver_x_body, qx_points = Visualizer.get_orientation_scatter(com, interval, 'x body', size, x_body,
                                                                      points_color=['black', 'yellow'], width=2)
        head = points_3D[:, -1, :]
        tail = points_3D[:, -2, :]
        dists = np.linalg.norm(head - tail, axis=1)
        y_body_center = tail + 0.65 * dists[:, np.newaxis] * x_body
        quiver_y_body, qy_points = Visualizer.get_orientation_scatter(y_body_center, interval, 'y body', size, y_body,
                                                                      points_color=['green', 'blue'])

        # Calculate the end points of the arrow
        arrow_start = y_body_center
        arrow_end = y_body_center + 1 * dists[:, np.newaxis] * facing

        # Create arrow trace
        arrow_trace = go.Scatter3d(x=[arrow_start[:, 0], arrow_end[:, 0]],
                                   y=[arrow_start[:, 1], arrow_end[:, 1]],
                                   z=[arrow_start[:, 2], arrow_end[:, 2]],
                                   mode='lines',
                                   name='Facing Arrow',
                                   line=dict(color='red', width=3))

        # Create a figure and add the traces
        fig = go.Figure(
            data=[trace_com, trace_left_tip, trace_right_tip, marker_start, marker_start_dark, marker_40_ms,
                  quiver_x_body, quiver_y_body, qx_points, qy_points, arrow_trace])

        # Update the scene
        scene = dict(camera=dict(eye=dict(x=1., y=1, z=1)),
                     xaxis=dict(nticks=10),
                     yaxis=dict(nticks=10),
                     zaxis=dict(nticks=10),
                     aspectmode='data'  # This makes the axes equal
                     )
        fig.update_scenes(scene)
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.01,
                font=dict(
                    size=16,  # This enlarges the labels
                    color="black"
                )
            )
        )
        # Write the figure to an HTML file
        pio.write_html(fig, save_path)

    @staticmethod
    def get_orientation_scatter(com, interval, name, size, x_body, points_color=['orange', 'blue'], width=5):
        x_quiver = []
        y_quiver = []
        z_quiver = []
        x_points = []
        y_points = []
        z_points = []
        marker_colors = []  # Initialize an empty list for individual marker colors

        for i in range(0, len(com), interval):
            start = [com[i, j] - size / 6 * x_body[i, j] for j in range(3)]
            end = [com[i, j] + size / 6 * x_body[i, j] for j in range(3)]
            x_quiver.extend([start[0], end[0], None])
            y_quiver.extend([start[1], end[1], None])
            z_quiver.extend([start[2], end[2], None])

            x_points.extend([start[0], end[0]])
            y_points.extend([start[1], end[1]])
            z_points.extend([start[2], end[2]])

            # Specify colors for each point: start point in orange, end point in blue
            marker_colors.extend(points_color)

        points = go.Scatter3d(x=x_points, y=y_points, z=z_points, mode='markers',
                              marker=dict(color=marker_colors, size=5),  # Apply the color array here
                              name=name + ' points')

        quiver_x_body = go.Scatter3d(x=x_quiver, y=y_quiver, z=z_quiver, mode='lines',
                                     line=dict(color='black', width=width),
                                     name=name)

        return quiver_x_body, points
    @staticmethod
    def plot_feature_with_plotly(input_hdf5_path, feature='roll_speed'):
        # Open the HDF5 file
        dir = os.path.dirname(input_hdf5_path)
        output_html_path = os.path.join(dir, f'{feature}.html')
        with h5py.File(input_hdf5_path, 'r') as hdf:
            # Prepare the plot
            fig = go.Figure()

            # Get all movie groups in the HDF5 file
            movies = list(hdf.keys())

            for movie in movies:
                group = hdf[movie]

                # Check if 'roll_angle', 'start_frame', and 'end_frame' datasets exist

                featrue_array = group[feature][:]
                first_analysed_frame = int(group['first_analysed_frame'][()])
                first_y_body_frame = int(group['first_y_body_frame'][()])
                start_frame = first_y_body_frame + first_analysed_frame
                end_frame = int(group['end_frame'][()])

                # Slice the roll_angle data to only between start and end frames
                if start_frame < end_frame and start_frame < len(featrue_array) and end_frame <= len(featrue_array):
                    roll_angle_segment = featrue_array[start_frame:end_frame]
                    x_values = np.arange(start_frame, end_frame) / 16

                    # Add a trace for each movie
                    fig.add_trace(go.Scatter(x=x_values, y=roll_angle_segment,
                                             mode='lines', name=f"Movie {movie}"))

            # Set plot layout
            fig.update_layout(
                title='Roll Angles Across Movies',
                xaxis_title='ms',
                yaxis_title='Roll Angle (degrees)',
                legend_title='Movies',
                template='plotly_white'
            )

            # Save the plot as HTML file
            fig.write_html(output_html_path)
            print(f"Plot saved to {output_html_path}")

    @staticmethod
    def get_data_from_h5(h5_path, data_name, frames=None):
        try:
            if frames is None:
                data = h5py.File(h5_path, 'r')[data_name][:]
            else:
                data = h5py.File(h5_path, 'r')[data_name][frames]
        except:
            data = h5py.File(h5_path, 'r')[data_name]
        return data

    @staticmethod
    def create_movie_mp4(h5_path_movie_path, mode=DISPLAY, save_frames=None,
                         reprojected_points_path=None, box_path=None,
                         save_path="movie_gif.gif",  rotate=False):
        # save_frames = np.arange(30, 130)
        zoom_factor = 1  # Adjust as needed
        points_2D = np.load(reprojected_points_path)
        first_analized_frame = Visualizer.get_data_from_h5(h5_path_movie_path, 'first_analysed_frame')[()]
        points_2D = add_nan_frames(points_2D, first_analized_frame)
        if save_frames is None:
            save_frames = np.arange(len(points_2D))

            save_frames = save_frames  # todo

            frames_from_box_and_reprojected_points = save_frames
            pass
        else:
            frames_from_box_and_reprojected_points = save_frames - first_analized_frame


        # points_2D = points_2D[frames_from_box_and_reprojected_points]
        channel_1 = [1, 1+3, 1+6, 1+9]
        # take the negative
        box = 1 - h5py.File(box_path, 'r')['/box'][frames_from_box_and_reprojected_points[:-first_analized_frame]][:, channel_1]
        box = add_nan_frames(box, first_analized_frame)
        # Assuming points is your (N, M, 3) array
        points = Visualizer.get_data_from_h5(h5_path_movie_path, 'points_3D')
        num_frames = len(points)
        # stroke planes
        my_stroke_planes = Visualizer.get_data_from_h5(h5_path_movie_path, 'stroke_planes')[:, :-1]
        # center of mass
        my_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'center_of_mass')
        # body vectors
        my_x_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'x_body')
        my_y_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'y_body')
        my_z_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'z_body')
        # wing vectors
        my_left_wing_span = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_span')
        my_right_wing_span = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_span')
        my_left_wing_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_chord')
        my_right_wing_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_chord')
        # wing tips
        my_left_wing_tip = Visualizer.get_data_from_h5(h5_path_movie_path, 'wings_tips_left')
        my_right_wing_tip = Visualizer.get_data_from_h5(h5_path_movie_path, 'wings_tips_right')
        # wings cm
        my_left_wing_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_CM')
        my_right_wing_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_CM')

        # Calculate the limits of the plot
        x_min = np.nanmin(points[save_frames, :, 0])
        y_min = np.nanmin(points[save_frames, :, 1])
        z_min = np.nanmin(points[save_frames, :, 2])

        x_max = np.nanmax(points[save_frames, :, 0])
        y_max = np.nanmax(points[save_frames, :, 1])
        z_max = np.nanmax(points[save_frames, :, 2])

        # Create a color array
        my_points_to_show = np.array([my_left_wing_CM, my_right_wing_CM, my_CM, my_left_wing_tip, my_right_wing_tip])
        num_points = points.shape[1]
        color_array = plt.cm.hsv(np.linspace(0, 1, num_points))

        fig = plt.figure(figsize=(35, 15))  # Adjust the figure size here
        fig.tight_layout()
        gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 8], height_ratios=[1, 1], wspace=0.01, hspace=0.01)
        # gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=0.1)  # Reduced spacing between subplots

        ax_2d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        for ax in ax_2d:
            ax.set_aspect('equal')
            ax.axis('off')  # Hide axes for a cleaner look
        ax_3d = fig.add_subplot(gs[:, 3], projection='3d')

        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])

        # Define the connections between points
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       (16, 17)]

        def add_quiver_axes(ax, origin, x_vec, y_vec, z_vec, scale=0.001, labels=None, color='r'):
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

        # Precompute the plane data
        def compute_plane_data(center, normal, y_body, size=0.005):
            d = size / 2

            # Ensure y_body is perpendicular to the normal
            y_body = y_body / np.linalg.norm(y_body)

            # Create a vector u that is perpendicular to both normal and y_body
            u = np.cross(normal, y_body)
            u = u / np.linalg.norm(u)

            # Calculate the corners of the plane
            corners = np.array([
                center + d * (u + y_body),
                center + d * (u - y_body),
                center + d * (-u - y_body),
                center + d * (-u + y_body),
            ])

            return corners

        plane_data = [compute_plane_data(my_CM[frame], my_stroke_planes[frame], my_y_body[frame]) for frame in
                      range(num_frames)]

        def plot_plane(ax, corners, color='g', alpha=0.2):
            vertices = [corners]
            plane = Poly3DCollection(vertices, color=color, alpha=alpha)
            ax.add_collection3d(plane)

        def update(frame, plot_all_my_points=True, labels=False):
            analysis_frame = frame + first_analized_frame
            print(f"frame : {analysis_frame}", flush=True)
            ax_3d.cla()  # Clear current axes
            if plot_all_my_points:
                for i in range(num_points):
                    if i not in [7, 15]:
                        ax_3d.scatter(points[analysis_frame, i, 0],
                                      points[analysis_frame, i, 1],
                                      points[analysis_frame, i, 2], color=color_array[i], s=5)
                for i, j in connections:
                    ax_3d.plot(points[analysis_frame, [i, j], 0],
                               points[analysis_frame, [i, j], 1],
                               points[analysis_frame, [i, j], 2], color='k',
                               linewidth=1)

            # Plot wing tips
            # ax_3d.scatter(my_left_wing_tip[analysis_frame, 0], my_left_wing_tip[analysis_frame, 1],
            #               my_left_wing_tip[analysis_frame, 2],
            #               color=color_array[0])
            # ax_3d.scatter(my_right_wing_tip[analysis_frame, 0], my_right_wing_tip[analysis_frame, 1],
            #               my_right_wing_tip[analysis_frame, 2],
            #               color=color_array[0])

            # Plot the center of mass
            my_cm_x, my_cm_y, my_cm_z = my_CM[analysis_frame]
            ax_3d.scatter(my_cm_x, my_cm_y, my_cm_z, color=color_array[0])

            # Add the stroke plane
            plot_plane(ax_3d, plane_data[analysis_frame])

            lables_xyz_body = ['Xb', 'Yb', 'Zb'] if labels else None
            labels_span_cord = ['Span', 'Chord'] if labels else None
            add_quiver_axes(ax_3d, (my_cm_x, my_cm_y, my_cm_z), my_x_body[analysis_frame], my_y_body[analysis_frame], my_z_body[analysis_frame],
                            color='r', labels=lables_xyz_body)
            add_quiver_axes(ax_3d, my_left_wing_CM[analysis_frame], my_left_wing_span[analysis_frame], my_left_wing_chord[analysis_frame], None,
                            color='r', labels=labels_span_cord)
            add_quiver_axes(ax_3d, my_right_wing_CM[analysis_frame], my_right_wing_span[analysis_frame], my_right_wing_chord[analysis_frame],
                            None, color='r', labels=labels_span_cord)

            try:
                zoom_scale = 0.003
                ax_3d.set_xlim([my_cm_x - zoom_scale, my_cm_x + zoom_scale])
                ax_3d.set_ylim([my_cm_y - zoom_scale, my_cm_y + zoom_scale])
                ax_3d.set_zlim([my_cm_z - zoom_scale, my_cm_z + zoom_scale])
            except:
                max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * zoom_factor
                mid_x = (x_max + x_min) / 2
                mid_y = (y_max + y_min) / 2
                mid_z = (z_max + z_min) / 2

                ax_3d.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
                ax_3d.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
                ax_3d.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

            ax_3d.set_box_aspect([1, 1, 1])
            if rotate:
                ax_3d.view_init(elev=20, azim=frame * 720 / num_frames)  # Adjusted to make rotation faster
            else:
                ax_3d.view_init(elev=20, azim=120)

            # Add these lines here
            from matplotlib.ticker import FuncFormatter

            def mm_formatter(x, pos):
                return f'{x * 1000:.0f}'

            ax_3d.set_xlabel('X (mm)')
            ax_3d.set_ylabel('Y (mm)')
            ax_3d.set_zlabel('Z (mm)')

            ax_3d.xaxis.set_major_formatter(FuncFormatter(mm_formatter))
            ax_3d.yaxis.set_major_formatter(FuncFormatter(mm_formatter))
            ax_3d.zaxis.set_major_formatter(FuncFormatter(mm_formatter))
            ax_3d.set_aspect('equal') # added this
            for i, ax in enumerate(ax_2d):
                analysis_frame = frame + first_analized_frame
                frame_in_box = analysis_frame + first_analized_frame
                print(frame, analysis_frame)
                ax.cla()
                image = box[analysis_frame, i].T
                head_tail_pnts = points_2D[analysis_frame, i, [-2, -1], :]
                cm = np.mean(head_tail_pnts, axis=0)
                shift_yx = np.array([192/2 - cm[1], 192/2 - cm[0]])
                image = scipy.ndimage.shift(image, shift_yx, cval=1, mode='constant')
                ax.imshow(image, cmap='gray', vmin=0, vmax=1)
                for j in range(num_points):
                    if j not in [7, 15]:
                        point = points_2D[analysis_frame, i, j, :]
                        point[0] += shift_yx[1]
                        point[1] += shift_yx[0]
                        ax.scatter(point[0],  point[1], color=color_array[j], s=3)
                # ax.scatter(cm[0] + shift_yx[1], cm[1] + shift_yx[0], c='blue')
                ax.axis('off')

            fig.canvas.draw_idle()

        # Create the animation

        if mode == SAVE:
            ani = animation.FuncAnimation(fig, update, frames=save_frames, interval=100)  # Adjust interval as needed

            if save_path:
                writer = animation.PillowWriter(fps=15)
                # writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
                ani.save(save_path, writer=writer)

            dir_path = os.path.dirname(save_path)
            gif_clip = VideoFileClip(save_path)
            rotated = "rotated" if rotate else "not rotated"
            save_path_mp4 = os.path.join(dir_path, f'analisys_mp4_{rotated}.mp4')
            gif_clip.write_videofile(save_path_mp4, codec="libx264")
            os.remove(save_path)
        elif mode == DISPLAY:
            # Add a slider below the 3D plot
            ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            slider = Slider(ax_slider, 'Frame', 0, len(save_frames) - 1, valinit=0, valfmt='%0.0f')

            def on_slider_update(val):
                frame = int(slider.val)
                update(frame)

            slider.on_changed(on_slider_update)

            # Key press event handler
            def on_key_press(event):
                current_val = slider.val
                if event.key == "right":
                    slider.set_val(min(current_val + 1, len(save_frames) - 1))
                elif event.key == "left":
                    slider.set_val(max(current_val - 1, 0))

            # Connect the key press event to the figure
            fig.canvas.mpl_connect("key_press_event", on_key_press)

            plt.show()

    @staticmethod
    def visualize_analisys_3D(h5_path_movie_path, ACTION=DISPLAY):
        zoom_factor = 1  # Adjust as needed

        # Load and scale all the data by 1000 to convert from meters to millimeters
        points = Visualizer.get_data_from_h5(h5_path_movie_path, 'points_3D') * 1000
        num_frames = len(points)

        # stroke planes (no scaling needed as these are vectors/directions)
        my_stroke_planes = Visualizer.get_data_from_h5(h5_path_movie_path, 'stroke_planes')[:, :-1]

        # Scale position data
        my_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'center_of_mass') * 1000

        # Body vectors (no scaling needed as these are unit vectors)
        my_x_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'x_body')
        my_y_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'y_body')
        my_z_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'z_body')

        # Wing vectors (no scaling needed as these are unit vectors)
        my_left_wing_span = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_span')
        my_right_wing_span = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_span')
        my_left_wing_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_chord')
        my_right_wing_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_chord')

        # Scale position data for wing tips
        my_left_wing_tip = Visualizer.get_data_from_h5(h5_path_movie_path, 'wings_tips_left') * 1000
        my_right_wing_tip = Visualizer.get_data_from_h5(h5_path_movie_path, 'wings_tips_right') * 1000

        # Scale position data for wings center of mass
        my_left_wing_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_CM') * 1000
        my_right_wing_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_CM') * 1000

        # omega body (no scaling needed as it's angular velocity)
        omega_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'omega_body')

        # Calculate the limits of the plot (now in millimeters)
        x_min = np.nanmin(points[:, :, 0])
        y_min = np.nanmin(points[:, :, 1])
        z_min = np.nanmin(points[:, :, 2])

        x_max = np.nanmax(points[:, :, 0])
        y_max = np.nanmax(points[:, :, 1])
        z_max = np.nanmax(points[:, :, 2])

        # Create a color array
        my_points_to_show = np.array([my_left_wing_CM, my_right_wing_CM, my_CM, my_left_wing_tip, my_right_wing_tip])
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, projection='3d')

        # Define the connections between points
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def add_quiver_axes(ax, origin, x_vec, y_vec, z_vec, scale=1, labels=None, color='r', fontsize=10):
            # Scale adjusted from 0.001 to 1 since we're now in millimeters
            x, y, z = origin
            if x_vec is not None:
                ax.quiver(x, y, z, x_vec[0], x_vec[1], x_vec[2], length=scale, color=color)
            if y_vec is not None:
                ax.quiver(x, y, z, y_vec[0], y_vec[1], y_vec[2], length=scale, color=color)
            if z_vec is not None:
                ax.quiver(x, y, z, z_vec[0], z_vec[1], z_vec[2], length=scale, color=color)

            # Adjust label placement scale
            scale = 1.35  # Adjusted from 0.00135
            if labels:
                if x_vec is not None:
                    ax.text(x + x_vec[0] * scale, y + x_vec[1] * scale, z + x_vec[2] * scale, labels[0], color=color,
                            fontsize=fontsize)
                if y_vec is not None:
                    ax.text(x + y_vec[0] * scale, y + y_vec[1] * scale, z + y_vec[2] * scale, labels[1], color=color,
                            fontsize=fontsize)
                if z_vec is not None:
                    ax.text(x + z_vec[0] * scale, y + z_vec[1] * scale, z + z_vec[2] * scale, labels[2], color=color,
                            fontsize=fontsize)

        def plot_plane(ax, corners, color='g', alpha=0.5):
            vertices = [corners]
            plane = Poly3DCollection(vertices, color=color, alpha=alpha)
            ax.add_collection3d(plane)

        def compute_plane_data(center, normal, y_body, size=5):
            # Adjusted size from 0.005 to 5 for millimeter scale
            d = size / 2

            # Ensure y_body is perpendicular to the normal
            y_body = y_body / np.linalg.norm(y_body)

            # Create a vector u that is perpendicular to both normal and y_body
            u = np.cross(normal, y_body)
            u = u / np.linalg.norm(u)

            # Calculate the corners of the plane
            corners = np.array([
                center + d * (u + y_body),
                center + d * (u - y_body),
                center + d * (-u - y_body),
                center + d * (-u + y_body),
            ])

            return corners

        plane_data = [compute_plane_data(my_CM[frame], my_stroke_planes[frame], my_y_body[frame])
                      for frame in range(num_frames)]

        def update(val, do_plot_plane=True, plot_wings=True):
            ax.cla()  # Clear current axes
            if val != -1:
                frame = val
            else:
                frame = int(slider.val)
            my_cm_x, my_cm_y, my_cm_z = my_CM[frame]
            if plot_wings:
                for i in range(num_points):
                    if i not in [7, 15]:
                        ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
                for i, j in connections:
                    ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')

                # Plot wing tips
                ax.scatter(my_left_wing_tip[frame, 0], my_left_wing_tip[frame, 1], my_left_wing_tip[frame, 2],
                           c=color_array[0])
                ax.scatter(my_right_wing_tip[frame, 0], my_right_wing_tip[frame, 1], my_right_wing_tip[frame, 2],
                           c=color_array[0])

                add_quiver_axes(ax, my_left_wing_CM[frame], my_left_wing_span[frame], my_left_wing_chord[frame],
                                None, color='r', labels=['Span', 'Chord'])
                add_quiver_axes(ax, my_right_wing_CM[frame], my_right_wing_span[frame], my_right_wing_chord[frame],
                                None, color='r', labels=['Span', 'Chord'])

            # Plot the center of mass
            ax.scatter(my_cm_x, my_cm_y, my_cm_z, c=color_array[0])
            if do_plot_plane:
                plot_plane(ax, plane_data[frame], alpha=0.2)

            add_quiver_axes(ax, (my_cm_x, my_cm_y, my_cm_z), my_x_body[frame], my_y_body[frame], my_z_body[frame],
                            color='r', labels=['Xb', 'Yb', 'Zb'])

            box_size = 0.0025 * 1000
            if ~np.isnan(my_cm_x):
                x_min, x_max = my_cm_x - box_size, my_cm_x + box_size
                y_min, y_max = my_cm_y - box_size, my_cm_y + box_size
                z_min, z_max = my_cm_z - box_size / 2, my_cm_z + box_size / 2
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_zlim(z_min, z_max)
                frame_points_projected = points[frame].copy()
                frame_points_projected[:, -1] = z_min
                # Draw projections
                for i in range(num_points):
                    if i not in [7, 15]:
                        ax.scatter(frame_points_projected[i, 0], frame_points_projected[i, 1],
                                      frame_points_projected[i, 2], color='gray', s=10)

                for i, j in connections:
                    ax.plot(frame_points_projected[[i, j], 0], frame_points_projected[[i, j], 1],
                               frame_points_projected[[i, j], 2], color='gray', linewidth=1)
                ax.plot([x_min - 0.2, x_max], [y_max, y_max], [z_min, z_min], 'k-',
                           linewidth=.5, clip_on=True)
                # ax_3d.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], 'k-',
                #            linewidth=.5, clip_on=False)
                ax.plot([x_max, x_max], [y_min - 0.2, y_max], [z_min, z_min], 'k-',
                           linewidth=.5, clip_on=True)

            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.set_box_aspect([1, 1, 1])
            ax.set_aspect('equal')
            fig.canvas.draw_idle()

        if ACTION == DISPLAY:
            slider.on_changed(update)

            def handle_key_event(event):
                if event.key == 'right':
                    slider.set_val(min(slider.val + 1, slider.valmax))
                elif event.key == 'left':
                    slider.set_val(max(slider.val - 1, slider.valmin))

            fig.canvas.mpl_connect('key_press_event', handle_key_event)
            update(-1)
            plt.show()
        else:
            ani = animation.FuncAnimation(fig, update, frames=save_frames, interval=100)
            dir_name = os.path.dirname(h5_path_movie_path)
            save_path = os.path.join(dir_name, "temp.gif")
            writer = animation.PillowWriter(fps=5)
            ani.save(save_path, writer=writer)
            dir_path = os.path.dirname(save_path)
            gif_clip = VideoFileClip(save_path)
            save_path_mp4 = os.path.join(dir_path, f'analisys_temp_mp4.mp4')
            gif_clip.write_videofile(save_path_mp4, codec="libx264")
            os.remove(save_path)


    @staticmethod
    def visualize_models_selection(all_models_combinations):
        model_names = ['left_wing_models', 'right_wing_models', 'head_tail_models', 'side_points_models']
        model_data = [all_models_combinations[i][..., :6] for i in range(len(all_models_combinations))]

        for model_name, models in zip(model_names, model_data):
            num_frames, num_models, num_candidates = models.shape

            # Set up the figure
            fig, ax = plt.subplots(5, 1, figsize=(16, 25))

            # Colors for plotting
            candidate_colors = plt.cm.get_cmap('tab10', num_candidates)
            model_colors = plt.cm.get_cmap('tab10', num_models)

            # Plot Candidates per Frame
            for candidate in range(num_candidates):
                frames = np.where(np.any(models[:, :, candidate], axis=1))[0]
                ax[0].scatter(frames, np.full_like(frames, candidate), color=candidate_colors(candidate),
                              label=f'Candidate {candidate + 1}', s=1)

            ax[0].set_xlabel('Frames')
            ax[0].set_ylabel('Candidates')
            ax[0].set_title('Candidates Chosen per Frame')
            ax[0].set_yticks(np.arange(num_candidates))
            ax[0].set_yticklabels([f'Candidate {i + 1}' for i in range(num_candidates)])
            ax[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))

            # Plot Models per Frame
            for model in range(num_models):
                frames = np.where(np.any(models[:, model, :], axis=1))[0]
                ax[1].scatter(frames, np.full_like(frames, model), color=model_colors(model),
                              label=f'Model {model + 1}',
                              s=1)

            ax[1].set_xlabel('Frames')
            ax[1].set_ylabel('Models')
            ax[1].set_title('Models Chosen per Frame')
            ax[1].set_yticks(np.arange(num_models))
            ax[1].set_yticklabels([f'Model {i + 1}' for i in range(num_models)])
            ax[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1))

            # Plot number of candidates chosen per frame
            candidates_per_frame = np.sum(np.any(models, axis=1), axis=1)
            ax[2].scatter(np.arange(num_frames), candidates_per_frame, color='blue', s=1)
            ax[2].set_xlabel('Frames')
            ax[2].set_ylabel('Number of Candidates')
            ax[2].set_title('Number of Candidates Chosen per Frame')

            # Plot number of models chosen per frame
            models_per_frame = np.sum(np.any(models, axis=2), axis=1)
            ax[3].scatter(np.arange(num_frames), models_per_frame, color='green', s=1)
            ax[3].set_xlabel('Frames')
            ax[3].set_ylabel('Number of Models')
            ax[3].set_title('Number of Models Chosen per Frame')

            # Plot Model Usage Histogram
            model_usage = np.sum(models, axis=(0, 2))
            ax[4].bar(np.arange(num_models), model_usage, color=[model_colors(i) for i in range(num_models)])
            ax[4].set_xlabel('Models')
            ax[4].set_ylabel('Usage Count')
            ax[4].set_title('Model Usage Histogram')
            ax[4].set_xticks(np.arange(num_models))
            ax[4].set_xticklabels([f'Model {i + 1}' for i in range(num_models)])

            # Display plots
            plt.tight_layout()
            plt.savefig(f'visualizations/{model_name}_visualized_models_selection.png')
            plt.close(fig)


    @staticmethod
    def plot_psi_or_theta_vs_phi(h5_path_movie_path, wing_bits=[0, 1, 2, 3, 4, 5, 6, 7, 8], wing='left', other_angle='psi'):
        # other angle is psi or theta
        fig = go.Figure()
        # Plot Pitch Body Angle
        for wing_bit in wing_bits:
            data_name = f"{wing}_full_wingbits/full_wingbit_{wing_bit}/phi_vals"
            phi = Visualizer.get_data_from_h5(h5_path_movie_path, data_name)
            data_name = f"{wing}_full_wingbits/full_wingbit_{wing_bit}/{other_angle}_vals"
            angle = np.squeeze(Visualizer.get_data_from_h5(h5_path_movie_path, data_name))
            fig.add_trace(go.Scatter(
                x=phi,
                y=angle,
                mode='lines',
                name=f'{other_angle} vs phi wingbit {wing_bit}',
                # line=dict(dash='dash')
            ))
        fig.update_layout(
            title=f'{wing} wing wingbits theta vs phi',
            xaxis_title='phi (degrees)',
            yaxis_title=f'{other_angle} (degrees)',
            legend_title='Legend',
            xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal axis scaling
            yaxis=dict(scaleanchor="x", scaleratio=1)  #
        )
        # Save the plot as an HTML file
        html_file_name = f'{other_angle} {wing} vs phi.html'
        dir = os.path.dirname(h5_path_movie_path)
        path = os.path.join(dir, html_file_name)
        fig.write_html(path)
        print(f'Saved: {path}')

    @staticmethod
    def compare_body_angles(h5_path_movie_path):
        angles_names = ['yaw', 'pitch', 'roll']
        # convert_to_ms = self.frame_rate / 1000
        yaw = Visualizer.get_data_from_h5(h5_path_movie_path, f'yaw_angle')
        pitch = Visualizer.get_data_from_h5(h5_path_movie_path, f'pitch_angle')
        roll = Visualizer.get_data_from_h5(h5_path_movie_path, f'roll_angle')
        angles = [yaw, pitch, roll]
        fig = go.Figure()
        for i in range(len(angles_names)):
            data = angles[i]
            angle = angles_names[i]
            fig.add_trace(go.Scatter(
                x=np.arange(len(data)),
                y=data,
                mode='lines',
                name=f'{angle.capitalize()}',
                # line=dict(dash='dash')
            ))
            fig.update_layout(
                title=f'{angle.capitalize()} Body Angle',
                xaxis_title='Frames',
                yaxis_title=f'{angle.capitalize()} Angle (degrees)',
                legend_title='Legend'
            )
        save_name = f'All body angles.html'
        dir = os.path.dirname(h5_path_movie_path)
        path = os.path.join(dir, save_name)
        fig.write_html(path)

    @staticmethod
    def display_omega_body(h5_path_movie_path):
        omega_body = Visualizer.get_data_from_h5(h5_path_movie_path, f'omega_body')
        fig = go.Figure()
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(go.Scatter(
                x=np.arange(len(omega_body)),
                y=omega_body[:, i],
                mode='lines',
                name=f'omega body {axis}',
            ))
        fig.update_layout(
            title=f'Omega body',
            xaxis_title='frames',
            yaxis_title=f'omega (degrees/sec^2)',
            legend_title='Legend'
        )
        save_name = f'omega_body.html'
        dir = os.path.dirname(h5_path_movie_path)
        path = os.path.join(dir, save_name)
        fig.write_html(path)

    @staticmethod
    def show_amplitude_graph(h5_path_movie_path, wing='left'):
        amplitudes = Visualizer.get_data_from_h5(h5_path_movie_path, f'{wing}_amplitudes')
        phi = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_phi_{wing}')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=amplitudes[:, 0],
            y=amplitudes[:, 1],
            mode='lines',
            name=f'{wing} wing amplitude',
            # line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(len(phi)),
            y=phi,
            mode='lines',
            name=f'{wing} wing phi',
            # line=dict(dash='dash')
        ))

        fig.update_layout(
            title=f'{wing} phi and amplitudes',
            xaxis_title='Frames',
            yaxis_title=f'amplitude and',
            legend_title='Legend'
        )
        save_name = f'{wing}_wing_phi_and_amplitude.html'
        dir = os.path.dirname(h5_path_movie_path)
        path = os.path.join(dir, save_name)
        fig.write_html(path)

    @staticmethod
    def show_left_vs_right_amplitude(h5_path_movie_path,):
        left_amplitudes = Visualizer.get_data_from_h5(h5_path_movie_path, f'left_amplitudes')
        right_amplitudes = Visualizer.get_data_from_h5(h5_path_movie_path, f'right_amplitudes')
        phi_left = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_phi_left')
        phi_right = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_phi_right')
        roll = Visualizer.get_data_from_h5(h5_path_movie_path, f'roll_angle')
        pitch = Visualizer.get_data_from_h5(h5_path_movie_path, f'pitch_angle')

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(len(phi_left)),
            y=pitch,
            mode='lines',
            name=f'pitch',
        ))

        fig.add_trace(go.Scatter(
            x=left_amplitudes[:, 0],
            y=left_amplitudes[:, 1] - right_amplitudes[:, 1],
            mode='lines',
            name=f'left minus right',
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(len(phi_left)),
            y=phi_left,
            mode='lines',
            name=f'phi left',
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(len(phi_right)),
            y=phi_right,
            mode='lines',
            name=f'phi right',
        ))

        fig.add_trace(go.Scatter(
            x=left_amplitudes[:, 0],
            y=left_amplitudes[:, 1],
            mode='lines',
            name=f'left wing amplitude',
        ))

        fig.add_trace(go.Scatter(
            x=right_amplitudes[:, 0],
            y=right_amplitudes[:, 1],
            mode='lines',
            name=f'right wing amplitude',
            # line=dict(dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(len(roll)),
            y=roll,
            mode='lines',
            name=f'roll',
            # line=dict(dash='dash')
        ))

        fig.update_layout(
            title=f'left vs right wing amplitudes',
            xaxis_title='Frames',
            yaxis_title=f'amplitude and',
            legend_title='Legend'
        )

        save_name = f'left vs right wing amplitudes.html'
        dir = os.path.dirname(h5_path_movie_path)
        path = os.path.join(dir, save_name)
        fig.write_html(path)

    @staticmethod
    def count_objects_in_group(h5_file_path, group_name):
        with h5py.File(h5_file_path, 'r') as h5_file:
            group = h5_file[group_name]
            num_objects = len(group.keys())
            return num_objects

    @staticmethod
    def plot_all_body_data(h5_path_movie_path):
        print(f"plot all body data for movie:\n{h5_path_movie_path}")
        # convert_to_ms = self.frame_rate / 1000
        yaw = Visualizer.get_data_from_h5(h5_path_movie_path, f'yaw_angle')
        pitch = Visualizer.get_data_from_h5(h5_path_movie_path, f'pitch_angle')
        roll = Visualizer.get_data_from_h5(h5_path_movie_path, f'roll_angle')

        yaw_dot = Visualizer.get_data_from_h5(h5_path_movie_path, 'yaw_dot')
        pitch_dot = Visualizer.get_data_from_h5(h5_path_movie_path, 'pitch_dot')
        roll_dot = Visualizer.get_data_from_h5(h5_path_movie_path, 'roll_dot')
        omega_body = Visualizer.get_data_from_h5(h5_path_movie_path, f'omega_body')

        left_amplitudes = Visualizer.get_data_from_h5(h5_path_movie_path, f'left_amplitudes')
        right_amplitudes = Visualizer.get_data_from_h5(h5_path_movie_path, f'right_amplitudes')

        phi_left = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_phi_left')
        phi_right = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_phi_right')

        psi_left = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_psi_left')
        psi_right = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_psi_right')

        theta_left = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_theta_left')
        theta_right = Visualizer.get_data_from_h5(h5_path_movie_path, f'wings_theta_right')

        # Example usage
        group_name = 'left_half_wingbits'
        num_left_half_wingbits = Visualizer.count_objects_in_group(h5_path_movie_path, group_name)
        group_name = 'right_half_wingbits'
        num_right_half_wingbits = Visualizer.count_objects_in_group(h5_path_movie_path, group_name)

        # num_halfbits = min(num_left_half_wingbits, num_right_half_wingbits)
        # mid_left = np.zeros((num_halfbits, 2))
        # mid_right = np.zeros((num_halfbits, 2))
        # for wing_bit in range(num_halfbits):
        #     data_name_left = f"left_half_wingbits/half_wingbit_{wing_bit}/avarage_value"
        #     left_val = Visualizer.get_data_from_h5(h5_path_movie_path, data_name_left)[()]
        #     data_name_right = f"right_half_wingbits/half_wingbit_{wing_bit}/avarage_value"
        #     right_val = Visualizer.get_data_from_h5(h5_path_movie_path, data_name_right)[()]
        #
        #     data_name_left = f"left_half_wingbits/half_wingbit_{wing_bit}/frames"
        #     left_frames = Visualizer.get_data_from_h5(h5_path_movie_path, data_name_left)[:]
        #     left_frame = left_frames[len(left_frames)//2]
        #
        #     data_name_right = f"right_half_wingbits/half_wingbit_{wing_bit}/frames"
        #     right_frames = Visualizer.get_data_from_h5(h5_path_movie_path, data_name_right)[:]
        #     right_frame = right_frames[len(right_frames)//2]
        #
        #     mid_left[wing_bit, 0], mid_left[wing_bit, 1] = left_frame, left_val
        #     mid_right[wing_bit, 0], mid_right[wing_bit, 1] = left_frame, right_val


        data = [yaw, pitch, roll,yaw_dot, pitch_dot, roll_dot, omega_body[:, 0], omega_body[:, 1], omega_body[:, 2],
                      phi_left, phi_right, psi_left, psi_right, theta_left, theta_right]
        angles_names = ['yaw', 'pitch', 'roll', 'yaw_dot', 'pitch_dot', 'roll_dot', 'omega_body_x', 'omega_body_y',
                        'omega_body_z', 'phi_left', 'phi_right', 'psi_left', 'psi_right', 'theta_left', 'theta_right']

        # 'normalize the data'
        # data = [data_i/np.nanmax(np.abs(data_i)) for data_i in data]

        fig = go.Figure()
        for i in range(len(angles_names)):
            data_i = data[i]
            angle = angles_names[i]
            fig.add_trace(go.Scatter(
                x=np.arange(len(data_i)),
                y=data_i,
                mode='lines',
                name=f'{angle.capitalize()}',
                # line=dict(dash='dash')
            ))

        fig.add_trace(go.Scatter(
            x=left_amplitudes[:, 0],
            y=left_amplitudes[:, 1],
            mode='lines',
            name=f'left wing amplitude',
        ))

        fig.add_trace(go.Scatter(
            x=right_amplitudes[:, 0],
            y=right_amplitudes[:, 1],
            mode='lines',
            name=f'right wing amplitude',
            # line=dict(dash='dash')
        ))

        # fig.add_trace(go.Scatter(
        #     x=left_amplitudes[:, 0],
        #     y=left_amplitudes[:, 1] - right_amplitudes[:, 1],
        #     mode='lines',
        #     name=f'left minus right',
        # ))

        # fig.add_trace(go.Scatter(
        #     x=mid_left[:, 0],
        #     y=mid_left[:, 1],
        #     mode='lines',
        #     name=f'left middle frame',
        # ))

        # fig.add_trace(go.Scatter(
        #     x=mid_right[:, 0],
        #     y=mid_right[:, 1],
        #     mode='lines',
        #     name=f'right middle frame',
        # ))

        fig.update_layout(
            title=f'All body data',
            xaxis_title='Frames',
            yaxis_title=f'normalized y',
            legend_title='Legend'
        )
        save_name = f'All body data.html'
        dir = os.path.dirname(h5_path_movie_path)
        path = os.path.join(dir, save_name)
        fig.write_html(path)

    @staticmethod
    def visualize_rotating_frames(x_frames, y_frames, z_frames, omega=None):
        def update(val):
            frame = int(slider.val)
            ax.cla()

            # Plot quivers
            if omega is not None:
                ax.quiver(0, 0, 0, omega[frame, 0], omega[frame, 1], omega[frame, 2], color='b')
            ax.quiver(0, 0, 0, x_frames[frame, 0], x_frames[frame, 1], x_frames[frame, 2], color='r')
            ax.quiver(0, 0, 0, y_frames[frame, 0], y_frames[frame, 1], y_frames[frame, 2], color='g')
            ax.quiver(0, 0, 0, z_frames[frame, 0], z_frames[frame, 1], z_frames[frame, 2], color='b')

            # Add labels
            if omega is not None:
                ax.text(omega[frame, 0], omega[frame, 1], omega[frame, 2], 'omega', color='b')
            ax.text(x_frames[frame, 0], x_frames[frame, 1], x_frames[frame, 2], 'x_body', color='r')
            ax.text(y_frames[frame, 0], y_frames[frame, 1], y_frames[frame, 2], 'y_body', color='g')
            ax.text(z_frames[frame, 0], z_frames[frame, 1], z_frames[frame, 2], 'z_body', color='b')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            fig.canvas.draw_idle()

        def on_key(event):
            frame = slider.val
            if event.key == 'left':
                frame -= 1
            elif event.key == 'right':
                frame += 1
            frame = np.clip(frame, 0, N - 1)
            slider.set_val(frame)

        N = len(x_frames)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initial plot
        ax.quiver(0, 0, 0, x_frames[0, 0], x_frames[0, 1], x_frames[0, 2], color='r')
        ax.quiver(0, 0, 0, y_frames[0, 0], y_frames[0, 1], y_frames[0, 2], color='g')
        ax.quiver(0, 0, 0, z_frames[0, 0], z_frames[0, 1], z_frames[0, 2], color='b')

        # Add labels
        ax.text(x_frames[0, 0], x_frames[0, 1], x_frames[0, 2], 'x', color='r')
        ax.text(y_frames[0, 0], y_frames[0, 1], y_frames[0, 2], 'y', color='g')
        ax.text(z_frames[0, 0], z_frames[0, 1], z_frames[0, 2], 'z', color='b')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Slider setup
        ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Frame', 0, N - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    @staticmethod
    def scale(img, scale_factor):
        N = img.shape[0]
        scale_factor = float(scale_factor)
        center = (N / 2, N / 2)
        zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale_factor)
        zoomed_img = cv2.warpAffine(img, zoom_matrix, (N, N), flags=cv2.INTER_CUBIC)
        return zoomed_img

    @staticmethod
    def visualized_channels_in_3D(fly_image, type_to_display):
        # Single image display
        # Normalize image for better visualization (optional)
        normalized_image = (fly_image - fly_image.min()) / (fly_image.max() - fly_image.min())

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Define the number of channels to display before and after '...'
        channels_before = 3  # Number of channels to display at the beginning
        channels_after = 2  # Number of channels to display at the end
        spacing = 100  # Spacing along the Z-axis

        max_channels = normalized_image.shape[-1]

        if type_to_display == OUTPUT:
            # Display the first few channels
            for i in range(channels_before):
                x = np.arange(fly_image.shape[1])
                y = np.arange(fly_image.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = np.full_like(X, spacing * i)  # Place each channel at different Z level

                # Plot the slice using surface
                channel = normalized_image[:, :, i]
                channel = Visualizer.scale(channel, scale_factor=1.3)
                channel = plt.cm.gray(channel)
                ax.plot_surface(X, Y, Z, facecolors=channel, rstride=1, cstride=1)

            # Add the vertical '...' operator
            ax.text(
                x=fly_image.shape[1] / 2,  # Centered in the X dimension
                y=fly_image.shape[0] / 2,  # Centered in the Y dimension
                z=spacing * (channels_before + 0.2),  # Position between displayed slices
                s='.\n.\n.\n.\n.\n.',  # Vertical dots
                fontsize=13,
                ha='center',
                va='center',
                color='black'
            )

            # Display the last few channels
            for i in range(channels_after):
                channel_idx = max_channels - channels_after + i
                x = np.arange(fly_image.shape[1])
                y = np.arange(fly_image.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = np.full_like(X, spacing * (channels_before + i + 1))  # Offset for spacing

                # Plot the slice using surface
                channel = normalized_image[:, :, channel_idx]
                channel = Visualizer.scale(channel, scale_factor=1.3)
                channel = plt.cm.gray(channel)
                ax.plot_surface(X, Y, Z, facecolors=channel, rstride=1, cstride=1)

        else:
            # Display all channels if total is less than or equal to (channels_before + channels_after)
            for i in range(max_channels):
                x = np.arange(fly_image.shape[1])
                y = np.arange(fly_image.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = np.full_like(X, spacing * i)  # Place each channel at different Z level

                # Plot the slice using surface
                channel = normalized_image[:, :, i]
                channel = Visualizer.scale(channel, scale_factor=1.3)
                channel = plt.cm.gray(channel)
                ax.plot_surface(X, Y, Z, facecolors=channel, rstride=1, cstride=1)

        # Customize the view angle
        ax.view_init(elev=24, azim=-60)
        type_txt = "Input of 3 Temporal Channels \nand 2 Binary Wings Masks" if normalized_image.shape[-1] < 6 else "Output of C Gaussian Heatmaps"

        # Label axes
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')

        # Remove Z-axis values
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Remove vertical grid lines
        ax.grid(False)
        ax.set_zlabel('Channel')
        ax.set_aspect('equal')
        ax.set_title(f'Pose Estimation CNN {type_txt}')
        plt.show()

    @staticmethod
    def get_contour_mask(binary_mask, thickness=1):
        # Convert the mask to an 8-bit single-channel image if it's not already
        binary_mask = binary_mask.astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_mask = np.zeros_like(binary_mask)

        # Draw contours on the empty mask
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=thickness)  # Draw filled contours

        return contour_mask

    @staticmethod
    def visualize_channels_all_in_one(fly_image, colored_heatmaps, camera):
        colored_heatmaps = Visualizer.scale(colored_heatmaps, scale_factor=1.3)
        fly_image[:, :, :3] = Visualizer.scale(fly_image[:, :, :3], 1.3)
        fly_image[:, :, 3:] = Visualizer.scale(fly_image[:, :, 3:], 1.3)
        left_mask_contour = Visualizer.get_contour_mask(fly_image[:, :, -2], thickness=2) // 255
        right_mask_contour = Visualizer.get_contour_mask(fly_image[:, :, -1], thickness=2) // 255
        time_channels = fly_image[:, :, :-2]
        time_channels[:, :, 0] += left_mask_contour
        time_channels[:, :, 2] += right_mask_contour
        # plt.imshow(right_mask_contour)
        # plt.show()
        time_channels =  np.clip(time_channels, 0, 1)
        colored_heatmaps = np.clip(colored_heatmaps, 0, 1)
        plt.imsave(f"time_channels_{camera}.png", time_channels)
        plt.imsave(f"colored_heatmaps_{camera}.png", colored_heatmaps)
        pass

    @staticmethod
    def load_autocorrelations(path, correlation_type):
        autocorrelations = []
        for dirpath, dirnames, _ in os.walk(path):
            for dirname in dirnames:
                if dirname.startswith('mov'):
                    h5_path = os.path.join(dirpath, dirname, f"{dirname}_analysis_smoothed.h5")
                    if os.path.isfile(h5_path):
                        try:
                            array = Visualizer.get_data_from_h5(h5_path, f'auto_correlation_{correlation_type}')
                        except:
                            a=0
                        autocorrelations.append(array)
        return autocorrelations

    @staticmethod
    def pad_arrays(data_list):
        # Determine the maximum length of the arrays in the list
        max_length = max(len(array) for array in data_list)

        # Pad each array with NaNs to this maximum length
        padded_arrays = np.array(
            [np.pad(array, (0, max_length - len(array)), 'constant', constant_values=np.nan) for array in data_list])
        return padded_arrays

    @staticmethod
    def plot_mean_std(data_cut, data_intact, title, filename, cut=700):
        fig = go.Figure()

        # Pad data arrays
        data_cut_padded = Visualizer.pad_arrays(data_cut)
        data_intact_padded = Visualizer.pad_arrays(data_intact)

        # Ensure all arrays are trimmed or extended to 'cut'
        data_cut_padded = data_cut_padded[:, :cut]
        data_intact_padded = data_intact_padded[:, :cut]

        # Calculate mean and standard deviation ignoring NaNs
        mean_cut = np.nanmean(data_cut_padded, axis=0)
        std_cut = np.nanstd(data_cut_padded, axis=0)
        mean_intact = np.nanmean(data_intact_padded, axis=0)
        std_intact = np.nanstd(data_intact_padded, axis=0)

        # Plotting mean and standard deviation
        x = np.arange(cut)
        fig.add_trace(go.Scatter(x=x, y=mean_cut, mode='lines', name='Mean Cut', line=dict(color='red')))
        fig.add_trace(
            go.Scatter(x=x, y=np.clip(mean_cut + std_cut, a_max=1, a_min=-1), fill=None, mode='lines', line=dict(color='red', dash='dash'),
                       showlegend=False))
        fig.add_trace(
            go.Scatter(x=x, y=mean_cut - std_cut, fill='tonexty', mode='lines', line=dict(color='red', dash='dash'),
                       name='Cut Std Dev'))

        fig.add_trace(go.Scatter(x=x, y=mean_intact, mode='lines', name='Mean Intact', line=dict(color='blue')))
        fig.add_trace(
            go.Scatter(x=x, y=np.clip(mean_intact + std_intact, a_max=1, a_min=-1), fill=None, mode='lines', line=dict(color='blue', dash='dash'),
                       showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mean_intact - std_intact, fill='tonexty', mode='lines',
                                 line=dict(color='blue', dash='dash'), name='Intact Std Dev'))

        # add to the plot a pvalue line for each T value (x axis)
        # for i in range(cut):
        #     t_stat, p_value = scipy.stats.ttest_ind(data_cut_padded[:, i], data_intact_padded[:, i],
        #                                             nan_policy='omit', equal_var=False)
        #     if p_value < 0.05:
        #         fig.add_shape(
        #             dict(type="line", x0=i, y0=-1, x1=i, y1=1, line=dict(color="green", width=1, dash="dashdot")))

        # Update layout
        fig.update_layout(title=title, xaxis_title="ΔT", yaxis_title="Autocorrelation")
        fig.write_html(filename)

        return fig

    @staticmethod
    def create_autocorrelation_plot(data_cut, data_intact, title, filename, cut=1500):
        fig = go.Figure()

        # Calculate and plot each 'cut' series
        auc_values = []
        for i, array in enumerate(data_cut):
            trimmed_array = array[:cut]
            fig.add_trace(go.Scatter(y=trimmed_array, mode='lines', name=f'cut {i + 1}', line=dict(color='red')))
            auc = 1 - simpson(trimmed_array, dx=1)
            auc_values.append(auc)

        # Calculate and plot each 'intact' series
        for i, array in enumerate(data_intact):
            trimmed_array = array[:cut]
            fig.add_trace(go.Scatter(y=trimmed_array, mode='lines', name=f'intact {i + 1}', line=dict(color='blue')))

        # Update layout and save
        fig.update_layout(title=title, xaxis_title="ΔT", yaxis_title="Autocorrelation")
        fig.write_html(filename)

    @staticmethod
    def visualize_auto_correlations(path_cut, path_intact, cut=900):
        xbody_cut = Visualizer.load_autocorrelations(path_cut, "x_body")
        xbody_intact = Visualizer.load_autocorrelations(path_intact, "x_body")
        Visualizer.create_autocorrelation_plot(xbody_cut, xbody_intact, "X-Body Autocorrelations",
                                               "xbody_autocorrelations.html", cut=cut)
        Visualizer.plot_mean_std(xbody_cut, xbody_intact, "X-Body Autocorrelations",
                                               "xbody_mean_std_autocorrelations.html", cut=cut)

        XYZ_cut = Visualizer.load_autocorrelations(path_cut, "axis_angle")
        XYZ_intact = Visualizer.load_autocorrelations(path_intact, "axis_angle")
        Visualizer.create_autocorrelation_plot(XYZ_cut, XYZ_intact, "XYZ Autocorrelations",
                                               "XYZ_autocorrelations.html", cut=cut)
        Visualizer.plot_mean_std(XYZ_cut, XYZ_intact, "Coordinate System Autocorrelations",
                                               "XYZ_mean_std_autocorrelations.html", cut=cut)

    @staticmethod
    def visualize_center_mass_speed_distributions(path_cut, path_intact):
        speeds = {"cut": [], "intact": []}
        for i, path in enumerate([path_cut, path_intact]):
            kind = "cut" if i == 0 and path else "intact"
            for dirpath, dirnames, _ in os.walk(path):
                for dirname in dirnames:
                    if dirname.startswith('mov'):
                        h5_path = os.path.join(dirpath, dirname, f"{dirname}_analysis_smoothed.h5")
                        if os.path.isfile(h5_path):
                            array = Visualizer.get_data_from_h5(h5_path, f'CM_speed')
                            speeds[kind].append(array)
        all_cut_speeds = np.concatenate(speeds["cut"])
        all_cut_speeds = all_cut_speeds[~np.isnan(all_cut_speeds)]
        all_intact_speeds = np.concatenate(speeds["intact"])
        all_intact_speeds = all_intact_speeds[~np.isnan(all_intact_speeds)]
        # Create a Plotly figure
        fig = go.Figure()

        # Add the histogram for "Cut" speeds
        fig.add_trace(go.Histogram(
            x=all_cut_speeds,
            nbinsx=30,
            histnorm='probability density',
            opacity=0.5,
            name="Cut",
            marker_color="red"
        ))

        # Add the histogram for "Intact" speeds
        fig.add_trace(go.Histogram(
            x=all_intact_speeds,
            nbinsx=30,
            histnorm='probability density',
            opacity=0.5,
            name="Intact",
            marker_color="blue"
        ))

        # Update layout for titles and axes
        fig.update_layout(
            title="Speed Distributions for Cut and Intact",
            xaxis_title="Speed",
            yaxis_title="Density",
            barmode="overlay"  # Overlay the histograms with transparency
        )

        # Save the plot to an HTML file
        fig.write_html("speed_distributions.html")

        # Display the plot (optional)
        fig.show()

    @staticmethod
    def visualized_channels_in_3D_axis(fly_image, type_to_display, ax):
        # Normalize image for better visualization (optional)
        normalized_image = (fly_image - fly_image.min()) / (fly_image.max() - fly_image.min())

        # Define the number of channels to display before and after '...'
        channels_before = 3  # Number of channels to display at the beginning
        channels_after = 2  # Number of channels to display at the end
        spacing = 100  # Spacing along the Z-axis

        max_channels = normalized_image.shape[-1]

        if max_channels > (channels_before + channels_after):
            # Display the first few channels
            for i in range(channels_before):
                x = np.arange(fly_image.shape[1])
                y = np.arange(fly_image.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = np.full_like(X, spacing * i)  # Place each channel at different Z level

                # Plot the slice using surface
                channel = normalized_image[:, :, i]
                channel = Visualizer.scale(channel, scale_factor=1.3)
                channel = plt.cm.gray(channel)
                ax.plot_surface(X, Y, Z, facecolors=channel, rstride=1, cstride=1)

            # Add the vertical '...' operator
            ax.text(
                x=fly_image.shape[1] / 2,  # Centered in the X dimension
                y=fly_image.shape[0] / 2,  # Centered in the Y dimension
                z=spacing * (channels_before + 0.2),  # Position between displayed slices
                s='.\n.\n.\n.\n.\n.',  # Vertical dots
                fontsize=13,
                ha='center',
                va='center',
                color='black'
            )

            # Display the last few channels
            for i in range(channels_after):
                channel_idx = max_channels - channels_after + i
                x = np.arange(fly_image.shape[1])
                y = np.arange(fly_image.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = np.full_like(X, spacing * (channels_before + i + 1))  # Offset for spacing

                # Plot the slice using surface
                channel = normalized_image[:, :, channel_idx]
                channel = Visualizer.scale(channel, scale_factor=1.3)
                channel = plt.cm.gray(channel)
                ax.plot_surface(X, Y, Z, facecolors=channel, rstride=1, cstride=1)

        else:
            # Display all channels if total is less than or equal to (channels_before + channels_after)
            for i in range(max_channels):
                x = np.arange(fly_image.shape[1])
                y = np.arange(fly_image.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = np.full_like(X, spacing * i)  # Place each channel at different Z level

                # Plot the slice using surface
                channel = normalized_image[:, :, i]
                channel = Visualizer.scale(channel, scale_factor=1.3)
                channel = plt.cm.gray(channel)
                ax.plot_surface(X, Y, Z, facecolors=channel, rstride=1, cstride=1)

        # Customize the view angle
        ax.view_init(elev=24, azim=-60)

        # Title based on type_to_display
        type_txt = "Input Channels (3 Temporal + 2 Masks)" if type_to_display == INPUT else "Output Channels (C Gaussian Heatmaps)"
        ax.set_title(type_txt, fontsize=20, pad=30)

        # Remove axis ticks and gridlines for clarity
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_aspect('equal')

    @staticmethod
    def visualize_analysis_3D_html(h5_path_movie_path, filename="3d_analysis_with_slider.html"):
        # Extract data from the HDF5 file
        points = Visualizer.get_data_from_h5(h5_path_movie_path, 'points_3D')
        points_for_scatter = np.delete(points, [7, 15], axis=1)
        random_colors = np.random.rand(points_for_scatter.shape[1], 3)  # Random RGB values for each point
        duration = 1
        points_size = int(np.nanmean(np.linalg.norm(points[:, -1] - points[:, -2], axis=1)) * 1000)
        opacity = 0.3
        font_size = points_size * 3
        line_width = points_size * 4
        num_frames = len(points)
        take_frames = np.arange(num_frames)
        take_frames = np.arange(500, 700)
        my_stroke_planes = Visualizer.get_data_from_h5(h5_path_movie_path, 'stroke_planes')[:, :-1]
        my_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'center_of_mass')
        my_x_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'x_body')
        my_y_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'y_body')
        my_z_body = Visualizer.get_data_from_h5(h5_path_movie_path, 'z_body')
        my_left_wing_span = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_span')
        my_right_wing_span = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_span')
        my_left_wing_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_chord')
        my_right_wing_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_chord')
        my_left_wing_tip = Visualizer.get_data_from_h5(h5_path_movie_path, 'wings_tips_left')
        my_right_wing_tip = Visualizer.get_data_from_h5(h5_path_movie_path, 'wings_tips_right')
        my_left_wing_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_CM')
        my_right_wing_CM = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_CM')

        # changed for this one
        my_left_wing_CM = points[:, 6, :]
        my_right_wing_CM = points[:, 6 + 8, :]

        left_lower_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'left_wing_lower_chord')
        right_lower_chord = Visualizer.get_data_from_h5(h5_path_movie_path, 'right_wing_lower_chord')

        lower_chord_points_left = points[:, 4, :]
        lower_chord_points_right = points[:, 4 + 8, :]
        distance_from_CM_left = np.linalg.norm(my_left_wing_CM - lower_chord_points_left, axis=-1)
        distance_from_CM_right = np.linalg.norm(my_right_wing_CM - lower_chord_points_right, axis=-1)

        distances_of_lower_chord_to_CM = [distance_from_CM_left, distance_from_CM_right]
        lower_chord_points = [lower_chord_points_left, lower_chord_points_right]
        lower_chords = [left_lower_chord, right_lower_chord]
        # Define connections between points
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       (16, 17)]

        # Helper function to compute the stroke plane vertices
        def compute_plane_vertices(center, normal, y_body, size=0.003):
            d = size / 2
            y_body = y_body / np.linalg.norm(y_body)
            u = np.cross(normal, y_body)
            u = u / np.linalg.norm(u)
            corners = np.array([
                center + d * (u + y_body),
                center + d * (u - y_body),
                center + d * (-u - y_body),
                center + d * (-u + y_body)
            ])
            return corners

        # Precompute stroke plane vertices for each frame
        plane_data = [compute_plane_vertices(my_CM[frame], my_stroke_planes[frame], my_y_body[frame]) for frame in
                      range(num_frames)]

        # Create the figure
        fig = go.Figure()

        # Create frames
        frames = []
        for frame in take_frames:
            frame_data = []

            # Define the center of mass for the origin
            origin = my_CM[frame]
            scale = np.linalg.norm(origin - points[frame, -1])  # Adjust the scale as needed


            # X-axis arrow
            frame_data.append(go.Scatter3d(
                x=[origin[0], origin[0] + my_x_body[frame][0] * scale],
                y=[origin[1], origin[1] + my_x_body[frame][1] * scale],
                z=[origin[2], origin[2] + my_x_body[frame][2] * scale],
                mode='lines+text',
                line=dict(color='red', width=line_width),
                text=['', 'Xb'],
                name='Xb',
                # textposition="top center",
                textfont=dict(size=font_size),
                showlegend=True
            ))

            # Y-axis arrow
            frame_data.append(go.Scatter3d(
                x=[origin[0], origin[0] + my_y_body[frame][0] * scale],
                y=[origin[1], origin[1] + my_y_body[frame][1] * scale],
                z=[origin[2], origin[2] + my_y_body[frame][2] * scale],
                mode='lines+text',
                line=dict(color='green', width=line_width),
                name='Yb',
                text=['', 'Yb'],
                # textposition="top center",
                textfont=dict(size=font_size),
                showlegend=True
            ))

            # Z-axis arrow
            frame_data.append(go.Scatter3d(
                x=[origin[0], origin[0] + my_z_body[frame][0] * scale],
                y=[origin[1], origin[1] + my_z_body[frame][1] * scale],
                z=[origin[2], origin[2] + my_z_body[frame][2] * scale],
                mode='lines+text',
                name='Zb',
                line=dict(color='blue', width=line_width),
                text=['', 'Zb'],
                textfont=dict(size=font_size),
                showlegend=True
            ))

            # Plot points and connections
            for i, j in connections:
                frame_data.append(go.Scatter3d(
                    x=[points[frame, i, 0], points[frame, j, 0]],
                    y=[points[frame, i, 1], points[frame, j, 1]],
                    z=[points[frame, i, 2], points[frame, j, 2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ))

            # # scatter the las center of mass points:
            # frame_data.append(go.Scatter3d(
            #     x=my_CM[frame-1000:frame, 0],
            #     y=my_CM[frame-1000:frame, 1],
            #     z=my_CM[frame-1000:frame, 2],
            #     mode='markers',
            #     marker=dict(
            #         size=points_size,
            #         color='purple',
            #         # Apply random RGB colors
            #     ),
            #     name='centers of mass'
            # ))

            # Generate random colors for each point in the scatter plot
            frame_data.append(go.Scatter3d(
                x=points_for_scatter[frame, :, 0],
                y=points_for_scatter[frame, :, 1],
                z=points_for_scatter[frame, :, 2],
                mode='markers',
                marker=dict(
                    size=points_size,
                    color=['rgb({}, {}, {})'.format(r * 255, g * 255, b * 255) for r, g, b in random_colors],
                    # Apply random RGB colors
                ),
                name='Points'
            ))

            # Plot center of mass
            frame_data.append(go.Scatter3d(
                x=[my_CM[frame, 0]],
                y=[my_CM[frame, 1]],
                z=[my_CM[frame, 2]],
                mode='markers',
                marker=dict(size=points_size, color='orange'),
                name='Center of Mass'
            ))

            # plot the upper and lower planes as meshes
            for plane_num in range(len(POINTS_4_PLANES)):
                color = 'blue' if plane_num % 2 == 0 else 'red'
                points_indices = POINTS_4_PLANES[plane_num]
                frame_data.append(go.Mesh3d(
                    x=points[frame, points_indices, 0],
                    y=points[frame, points_indices, 1],
                    z=points[frame, points_indices, 2],
                    color=color,
                    opacity=opacity,
                    name='wing plane',
                    showlegend=True,
                ))
            # plot boundaries
            boundarie_points = np.array([np.array(coord) for coord in itertools.product([-1, 1], repeat=3)])
            boundarie_points = 0.003 * boundarie_points + my_CM[frame]
            for i in range(boundarie_points.shape[0]):
                frame_data.append(go.Scatter3d(
                    x=[boundarie_points[i, 0]],
                    y=[boundarie_points[i, 1]],
                    z=[boundarie_points[i, 2]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=1, color='white'),
                ))

            # Plot span and chord vectors as quivers
            left_vec_size = np.linalg.norm(my_left_wing_CM[frame] - my_left_wing_tip[frame])
            right_vec_size = np.linalg.norm(my_right_wing_CM[frame] - my_right_wing_tip[frame])
            for wing_CM, wing_span, wing_chord, vec_size in zip(
                    [my_left_wing_CM[frame], my_right_wing_CM[frame]],
                    [my_left_wing_span[frame], my_right_wing_span[frame]],
                    [my_left_wing_chord[frame], my_right_wing_chord[frame]],
                    [left_vec_size, right_vec_size]):
                frame_data.extend([
                    go.Scatter3d(
                        x=[wing_CM[0], wing_CM[0] + wing_span[0] * vec_size],
                        y=[wing_CM[1], wing_CM[1] + wing_span[1] * vec_size],
                        z=[wing_CM[2], wing_CM[2] + wing_span[2] * vec_size],
                        mode='lines+text',
                        line=dict(color='red', width=4),
                        text=['', 'span'],
                        textfont=dict(size=font_size),
                        name="span",
                        showlegend=True,
                    ),
                    go.Scatter3d(
                        x=[wing_CM[0], wing_CM[0] + wing_chord[0] * vec_size],
                        y=[wing_CM[1], wing_CM[1] + wing_chord[1] * vec_size],
                        z=[wing_CM[2], wing_CM[2] + wing_chord[2] * vec_size],
                        mode='lines+text',
                        line=dict(color='blue', width=4),
                        text=['', 'chord'],
                        textfont=dict(size=font_size),
                        name="chord",
                        showlegend=True,
                    )
                ])
            for wing in range(2):
                lower_chord_point = lower_chord_points[wing][frame]
                lower_chord_vec = lower_chords[wing][frame]
                distance = distances_of_lower_chord_to_CM[wing][frame]
                frame_data.extend([go.Scatter3d(
                    x=[lower_chord_point[0], lower_chord_point[0] + lower_chord_vec[0] * distance],
                    y=[lower_chord_point[1], lower_chord_point[1] + lower_chord_vec[1] * distance],
                    z=[lower_chord_point[2], lower_chord_point[2] + lower_chord_vec[2] * distance],
                    mode='lines+text',
                    line=dict(color='green', width=4),
                    text=['', 'lower chord'],
                    textfont=dict(size=font_size),
                    name="lower chord",
                    showlegend=True,
                )])

            # Plot stroke plane
            stroke_plane_corners = plane_data[frame]
            frame_data.append(go.Mesh3d(
                x=stroke_plane_corners[:, 0],
                y=stroke_plane_corners[:, 1],
                z=stroke_plane_corners[:, 2],
                color='green',
                opacity=opacity,
                name='Stroke Plane',
                showlegend=True,
            ))

            frames.append(go.Frame(data=frame_data, name=str(frame)))

        # Add initial frame data
        fig.add_traces(frames[0].data)

        # Set up the slider
        sliders = [dict(
            steps=[dict(method='animate', args=[[str(frame)],
                                                dict(mode='immediate', frame=dict(duration=duration, redraw=True),
                                                     transition=dict(duration=0))],
                        label=str(frame)) for frame in take_frames],
            active=0,
            transition=dict(duration=0),
            x=0.1, y=0,
            currentvalue=dict(font=dict(size=20), prefix='Frame: ', visible=True, xanchor='center'),
            len=0.9
        )]

        # Update layout with equal aspect ratio and slider
        fig.update(frames=frames)
        fig.update_layout(
            scene=dict(
                camera=dict(eye=dict(x=1.0, y=1.0, z=1.0)),
                xaxis=dict(nticks=10),
                yaxis=dict(nticks=10),
                zaxis=dict(nticks=10),
                aspectmode='data',  # Equal scaling across axes
                dragmode='orbit'  # Enable mouse rotation (orbit mode)
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, dict(frame=dict(duration=duration, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='Pause', method='animate',
                         args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
                ],
                x=0.1, y=0.05
            )],
            sliders=sliders
        )

        # Save the figure to HTML
        fig.write_html(filename)

    @staticmethod
    def visualize_3D_point_selection(movie_dir_path):
        _, all_points_list = predict_3D_points_all_pairs(movie_dir_path)
        all_points_list = [all_points_list[i][:, :, :6, :] for i in range(len(all_points_list))]
        all_points_array = np.concatenate(all_points_list, axis=2)
        final_points = np.load(os.path.join(movie_dir_path, 'points_3D_ensemble_best_method.npy'))
        point_num = -2
        axis = 0
        frames = np.arange(1000, 1200)
        selected = final_points[frames, point_num, axis]
        candidates = all_points_array[frames, point_num, :, axis]
        plt.plot(selected)
        plt.plot(candidates, '.', markersize=2)
        plt.show()

    @staticmethod
    def load_all_attributes_from_h5(path, attribute):
        attributes = []
        for dirpath, dirnames, _ in os.walk(path):
            for dirname in dirnames:
                if dirname.startswith('mov'):
                    h5_path = os.path.join(dirpath, dirname, f"{dirname}_analysis_smoothed.h5")
                    if os.path.isfile(h5_path):
                        try:
                            array = Visualizer.get_data_from_h5(h5_path, attribute)
                        except:
                            a = 0
                        attributes.append(array)
        return attributes

    @staticmethod
    def visualize_speeds_distributions(path_cut, path_intact):
        all_speeds_cut = Visualizer.load_all_attributes_from_h5(path_cut, attribute='CM_speed')
        all_speeds_intact = Visualizer.load_all_attributes_from_h5(path_intact, attribute='CM_speed')
        Visualizer.display_speeds(all_speeds_cut, all_speeds_intact)

    @staticmethod
    def display_speeds(all_speeds_cut, all_speeds_intact, kind="flies", nbins=20):
        names = ["cut", "intact"]
        speeds = [all_speeds_cut, all_speeds_intact]

        # Compute the combined dataset to determine common bin edges
        all_data = np.concatenate([np.concatenate(speed) for speed in speeds])
        cleaned_all_data = all_data[~np.isnan(all_data)]

        # Determine bin edges
        bins = np.histogram_bin_edges(cleaned_all_data, bins=nbins)

        plt.figure()
        for i in range(2):
            speed = speeds[i]
            name = names[i]
            concatenated_vector = np.concatenate(speed)
            cleaned_vector = concatenated_vector[~np.isnan(concatenated_vector)]

            # Calculate the histogram counts (not density)
            counts, _ = np.histogram(cleaned_vector, bins=bins)

            # Convert counts to probabilities by dividing by total number of samples
            probabilities = counts / len(cleaned_vector)

            # Print the probabilities for each bin
            print(f"Probabilities for {name}: {probabilities}")

            # Calculate the mean and standard deviation
            mean_speed = np.mean(cleaned_vector)
            std_speed = np.std(cleaned_vector)

            # Update the label with mean and std
            label = f"{name} (mean: {mean_speed:.2f} [m/s] | std: {std_speed:.2f} [m/s])"

            # Plot histogram with actual probabilities (not density)
            plt.hist(cleaned_vector, bins=bins, edgecolor='black', alpha=0.75,
                     label=label, weights=np.ones_like(cleaned_vector) / len(cleaned_vector))

        plt.legend()
        plt.title("Distribution of Ground Speed of the Center of Mass")
        plt.xlabel("[m/s]")
        plt.ylabel("Probability")
        plt.savefig(f'{kind}_speeds_distributions.png', dpi=600)

    @staticmethod
    def visualize_monte_carlo_3d(points_data, filename="monte_carlo_sampling_3d.html"):

        # Convert list to numpy array if necessary
        points = np.array(points_data)[:, 10:20]
        head_tail = np.median(points[:, :, [-2, -1], :], axis=0)
        median_points = np.median(points, axis=0)
        CM = np.mean(head_tail, axis=1)
        num_samples, num_frames, num_points, _ = points.shape

        random_colors =  np.random.rand(15, 3)

        # Generate distinct colors for each point (P)
        point_colors = np.random.rand(num_points, 3)

        # Create figure
        fig = go.Figure()

        # Create frames
        frames = []
        for frame in range(num_frames):
            frame_data = []

            for i, j in connections:
                frame_data.append(go.Scatter3d(
                    x=[median_points[frame, i, 0], median_points[frame, j, 0]],
                    y=[median_points[frame, i, 1], median_points[frame, j, 1]],
                    z=[median_points[frame, i, 2], median_points[frame, j, 2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ))
            # plot boundaries
            boundarie_points = np.array([np.array(coord) for coord in itertools.product([-1, 1], repeat=3)])
            boundarie_points = 0.003 * boundarie_points + CM[frame]
            for i in range(boundarie_points.shape[0]):
                frame_data.append(go.Scatter3d(
                    x=[boundarie_points[i, 0]],
                    y=[boundarie_points[i, 1]],
                    z=[boundarie_points[i, 2]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=1, color='white'),
                ))

            # For each point P, plot all Monte Carlo samples
            for p in range(num_points):
                # Get all Monte Carlo samples for this point at this frame
                if p not in [7, 15]:
                    point_samples = points[:, frame, p, :]  # Shape: (M, 3)

                    # Create scatter plot for this point's Monte Carlo samples
                    frame_data.append(go.Scatter3d(
                        x=point_samples[:, 0],
                        y=point_samples[:, 1],
                        z=point_samples[:, 2],
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=f'rgb({point_colors[p, 0] * 255}, {point_colors[p, 1] * 255}, {point_colors[p, 2] * 255})',
                        ),
                        name=f'Point {p}',
                        showlegend=(frame == 0)  # Only show legend for first frame
                    ))

            frames.append(go.Frame(data=frame_data, name=str(frame)))

        # Add initial frame data
        fig.add_traces(frames[0].data)

        # Set up slider
        sliders = [dict(
            steps=[
                dict(
                    method='animate',
                    args=[[str(frame)],
                          dict(mode='immediate',
                               frame=dict(duration=100, redraw=True),
                               transition=dict(duration=0))],
                    label=str(frame)
                )
                for frame in range(num_frames)
            ],
            active=0,
            transition=dict(duration=0),
            x=0.1,
            y=0,
            currentvalue=dict(
                font=dict(size=12),
                prefix='Frame: ',
                visible=True,
                xanchor='center'
            ),
            len=0.9
        )]

        # Update layout
        fig.update(frames=frames)
        fig.update_layout(
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='data',  # Equal scaling across axes
                dragmode='orbit'  # Enable mouse rotation
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=100, redraw=True),
                                          fromcurrent=True,
                                          mode='immediate')]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(mode='immediate',
                                            frame=dict(duration=0, redraw=True))])
                ],
                x=0.1,
                y=0.05
            )],
            sliders=sliders
        )

        # Save to HTML file
        fig.write_html(filename)

        return fig

    @staticmethod
    def visualize_points_and_images(h5_path, reprojected_points_path, box_path):
        """
        Visualize 3D points and 2D images with their corresponding points, with interactive slider and keyboard controls.

        Args:
            h5_path: Path to the H5 file containing 3D points
            reprojected_points_path: Path to the reprojected 2D points
            box_path: Path to the box data
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import h5py
        import scipy.ndimage

        # Load 2D points and add nan frames
        points_2D = np.load(reprojected_points_path)
        first_analized_frame = Visualizer.get_data_from_h5(h5_path, 'first_analysed_frame')[()]
        points_2D = add_nan_frames(points_2D, first_analized_frame)

        # Load box data
        channel_1 = [1, 1 + 3, 1 + 6, 1 + 9]
        box = 1 - h5py.File(box_path, 'r')['/box'][:, channel_1]
        box = add_nan_frames(box, first_analized_frame)

        # Load 3D points
        points = Visualizer.get_data_from_h5(h5_path, 'points_3D')
        num_frames = len(points)

        # Calculate global limits for 3D plot
        mask = ~np.isnan(points)
        if np.any(mask):
            global_min = np.array([
                np.min(points[:, :, 0][mask[:, :, 0]]),
                np.min(points[:, :, 1][mask[:, :, 1]]),
                np.min(points[:, :, 2][mask[:, :, 2]])
            ])
            global_max = np.array([
                np.max(points[:, :, 0][mask[:, :, 0]]),
                np.max(points[:, :, 1][mask[:, :, 1]]),
                np.max(points[:, :, 2][mask[:, :, 2]])
            ])
            global_center = (global_max + global_min) / 2
            global_range = np.max(global_max - global_min)
        else:
            global_center = np.array([0., 0., 0.])
            global_range = 0.01  # fallback value

        # Set up the figure with minimal spacing
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig, width_ratios=[2, 2, 6], height_ratios=[1, 1])
        gs.update(wspace=0.0, hspace=0.0)

        # Create 2D axes (2x2 grid) and 3D axis
        ax_2d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        ax_3d = fig.add_subplot(gs[:, 2], projection='3d')

        # Remove margins but keep a small bottom margin for the slider
        plt.subplots_adjust(left=0, right=1, bottom=0.05, top=1)

        # Create slider
        ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%0.0f')

        # Set up 2D axes properties
        for ax in ax_2d:
            ax.set_aspect('equal')
            ax.axis('off')

        # Create color array for points
        num_points = points.shape[1]
        color_array = plt.cm.hsv(np.linspace(0, 1, num_points))

        # Define connections between points
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       (16, 17)]

        def update_plot(frame_number):
            # Clear previous plots
            ax_3d.cla()
            for ax in ax_2d:
                ax.cla()

            # Get frame points and valid mask
            frame_points = points[frame_number]
            valid_mask = ~np.isnan(frame_points).any(axis=1)

            # Calculate center of mass
            if np.any(valid_mask):
                center_of_mass = np.mean(frame_points[[-1, -2]], axis=0)
            else:
                center_of_mass = np.array([0., 0., 0.])

            # Set box dimensions
            box_size = 0.0025
            x_min, x_max = center_of_mass[0] - box_size, center_of_mass[0] + box_size
            y_min, y_max = center_of_mass[1] - box_size, center_of_mass[1] + box_size
            z_min, z_max = center_of_mass[2] - box_size / 2, center_of_mass[2] + box_size / 2

            # Create projected points
            frame_points_projected = frame_points.copy()
            frame_points_projected[:, -1] = z_min

            # Set up 3D axis with exact limits
            ax_3d.set_autoscale_on(False)  # Disable autoscaling
            ax_3d.autoscale(False)
            ax_3d.margins(x=0, y=0, z=0)
            ax_3d.set_box_aspect([1, 1, 1])
            ax_3d.view_init(elev=20, azim=120)

            # Draw points and connections
            for i in range(num_points):
                if i not in [7, 15] and valid_mask[i]:
                    ax_3d.scatter(frame_points[i, 0], frame_points[i, 1], frame_points[i, 2],
                                  color=color_array[i], s=15)

            for i, j in connections:
                if valid_mask[i] and valid_mask[j]:
                    ax_3d.plot(frame_points[[i, j], 0], frame_points[[i, j], 1],
                               frame_points[[i, j], 2], color='k', linewidth=1)

            # Draw projections
            for i in range(num_points):
                if i not in [7, 15] and valid_mask[i]:
                    ax_3d.scatter(frame_points_projected[i, 0], frame_points_projected[i, 1],
                                  frame_points_projected[i, 2], color='gray', s=5)

            for i, j in connections:
                if valid_mask[i] and valid_mask[j]:
                    ax_3d.plot(frame_points_projected[[i, j], 0], frame_points_projected[[i, j], 1],
                               frame_points_projected[[i, j], 2], color='gray', linewidth=1)

            # Draw box edges using exact limits and clip_on=False to prevent edge clipping
            # Bottom edges
            # ax_3d.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], 'k-',
            #            linewidth=.5, clip_on=False)
            ax_3d.plot([x_min - 0.0002, x_max], [y_max, y_max], [z_min, z_min], 'k-',
                       linewidth=.5, clip_on=True)
            # ax_3d.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], 'k-',
            #            linewidth=.5, clip_on=False)
            ax_3d.plot([x_max, x_max], [y_min - 0.0002, y_max], [z_min, z_min], 'k-',
                       linewidth=.5, clip_on=True)

            # Reset limits after drawing everything
            ax_3d.set_xlim(x_min + 0.00001, x_max - 0.00001)
            ax_3d.set_ylim(y_min + 0.00001, y_max - 0.00001)
            ax_3d.set_zlim(z_min, z_max)
            ax_3d.margins(x=0, y=0, z=0)

            # Format axes
            def mm_formatter(x, pos):
                return f'{x * 1000:.0f}'

            ax_3d.set_xlabel('X (mm)')
            ax_3d.set_ylabel('Y (mm)')
            ax_3d.set_zlabel('Z (mm)')
            ax_3d.xaxis.set_major_formatter(plt.FuncFormatter(mm_formatter))
            ax_3d.yaxis.set_major_formatter(plt.FuncFormatter(mm_formatter))
            ax_3d.zaxis.set_major_formatter(plt.FuncFormatter(mm_formatter))
            ax_3d.set_aspect('equal')

            # Plot 2D images and points
            for i, ax in enumerate(ax_2d):
                try:
                    image = box[frame_number, i].T
                    valid_points_2d = points_2D[frame_number, i, [-2, -1], :]
                    if not np.isnan(valid_points_2d).any():
                        cm = np.mean(valid_points_2d, axis=0)
                        shift_yx = np.array([192 / 2 - cm[1], 192 / 2 - cm[0]])
                        image = scipy.ndimage.shift(image, shift_yx, cval=1, mode='constant')
                    ax.imshow(image, cmap='gray', vmin=0, vmax=1)

                    # Add camera label
                    ax.text(96, 20, f"Camera {i + 1}", color='white', fontsize=10,
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2),
                            ha='center', va='center')

                    # Plot 2D points
                    for j in range(num_points):
                        if j not in [7, 15]:
                            point = points_2D[frame_number, i, j, :]
                            if not np.isnan(point).any():
                                point[0] += shift_yx[1]
                                point[1] += shift_yx[0]
                                ax.scatter(point[0], point[1], color=color_array[j], s=10)
                except Exception as e:
                    print(f"Error plotting 2D image {i}: {e}")
                ax.axis('off')

            fig.canvas.draw_idle()

        # Set up slider callback
        def on_slider_changed(val):
            frame_number = int(slider.val)
            update_plot(frame_number)

        slider.on_changed(on_slider_changed)

        # Set up keyboard controls
        def on_key_press(event):
            if event.key == 'left':
                current_frame = int(slider.val)
                new_frame = max(current_frame - 1, 0)
                slider.set_val(new_frame)
            elif event.key == 'right':
                current_frame = int(slider.val)
                new_frame = min(current_frame + 1, num_frames - 1)
                slider.set_val(new_frame)

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Initial plot
        update_plot(0)
        plt.show()

    @staticmethod
    def input_to_output():
        # Define paths to the images
        path_image1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\input image example.png"  # Path to the raw movie input
        path_image2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\time_channels_2.png"  # Path to the CNN input (5 channels)
        path_image3 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\colored heatmaps.png" # Path to the CNN output (C channels)

        # Load the images
        image1 = cv2.imread(path_image1, cv2.IMREAD_GRAYSCALE)  # Raw movie input as grayscale
        image2 = cv2.imread(path_image2, cv2.IMREAD_UNCHANGED)  # CNN input, preserve all channels
        image3 = cv2.imread(path_image3, cv2.IMREAD_UNCHANGED)  # CNN output, preserve all channels

        # Verify loading and provide feedback
        if image1 is None or image2 is None or image3 is None:
            raise FileNotFoundError("One or more images could not be loaded. Check file paths.")

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the raw movie input
        axes[0].imshow(image1, cmap='gray' if len(image1.shape) == 2 else None)
        axes[0].set_title('Raw Movie Input', fontsize=20)
        axes[0].axis('off')

        # Plot the CNN input
        axes[1].imshow(image2, cmap=None)
        axes[1].set_title('CNN Input', fontsize=20)
        axes[1].axis('off')

        # Plot the CNN output
        axes[2].imshow(image3, cmap=None)
        axes[2].set_title('CNN Output', fontsize=20)
        axes[2].axis('off')

        # Annotate the relationships
        fig.suptitle('Illustration of CNN Input/Output Flow', fontsize=25)

        # Show the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to accommodate the title
        plt.show()


def create_gif_one_movie(base_path, mov_num, box_path, rotate=False):
    h5_path_movie_path = os.path.join(base_path, f'mov{mov_num}_analysis_smoothed.h5')
    reprojected_points_path = os.path.join(base_path, 'points_ensemble_smoothed_reprojected.npy')
    save_frames = None
    dir = os.path.dirname(reprojected_points_path)
    save_path_gif_rotated = os.path.join(dir, 'analisys_rotated.gif')
    Visualizer.create_movie_mp4(h5_path_movie_path, save_frames=save_frames,
                                reprojected_points_path=reprojected_points_path,
                                box_path=box_path, save_path=save_path_gif_rotated, rotate=rotate)


def get_movie_h5(movie_dir_path):
    # Get the list of files in the directory
    files = os.listdir(movie_dir_path)

    # Define the regex pattern
    pattern = re.compile(r'^movie.*\.h5$')

    # Search for the file
    for file in files:
        if pattern.match(file):
            print("Found file:", file)
            return os.path.join(movie_dir_path, file)
    else:
        print("No matching file found.")
    return None


def create_mp4_directory(base_path):
    movies = [dir for dir in os.listdir(base_path) if dir.startswith('mov')]
    for movie in movies:
        movie_dir_path = os.path.join(base_path, movie)
        match = re.search(r'\d+', movie)
        mov_num = int(match.group())

        started_file_path = os.path.join(movie_dir_path, 'started_mp4.txt')
        if not os.path.exists(started_file_path):
            with open(started_file_path, 'w') as file:
                file.write('Processing started')
        else:
            print(f"Skipping {movie_dir_path}, processing already started.", flush=True)
            continue

        done_file_path = os.path.join(movie_dir_path, 'done_mp4.txt')
        if os.path.exists(done_file_path):
            print(f"Skipping {movie_dir_path}, already processed.")
            continue

        box_path = get_movie_h5(movie_dir_path)
        try:
            print(f"doing movie {mov_num}", flush=True)
            create_gif_one_movie(movie_dir_path, mov_num, box_path)
        except Exception as e:
            print(f"movie {movie} returned this exception:")
            print(e)


def traverse_and_plot(directory_path):
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.endswith('analysis_smoothed.h5'):
                        h5_file_path = os.path.join(root, file)
                        try:
                            Visualizer.plot_all_body_data(h5_file_path)
                            print(f"finished movie {file}")
                        except Exception as e:
                            print(f"exception {e} occured during the opening og file {file}")


def visualize_heatmaps(image):
    from matplotlib.colors import Normalize
    """
    Visualize the heatmaps from an image of shape (192, 192, C),
    where C is the number of channels.
    """
    # Ensure the input image has 3 dimensions
    if len(image.shape) != 3:
        raise ValueError("Input image must have 3 dimensions (H, W, C)")

    h, w, c = image.shape

    # Initialize the final composite image
    composite = np.zeros((h, w, 3), dtype=np.float32)

    # Predefined colormaps
    colormaps = [
        'Reds', 'Blues', 'Greens', 'Oranges',
        'Purples', 'pink', 'YlGn', 'cool'
    ]

    # Loop through each channel and add the heatmap to the composite image
    for i in range(c):
        if i == 7 or i == 15:
            continue
        heatmap = image[:, :, i]

        # Normalize the heatmap
        norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max(), clip=True)
        heatmap_norm = norm(heatmap)

        # Apply the colormap
        colormap = plt.get_cmap(colormaps[i % len(colormaps)])
        colored_heatmap = heatmap[:, :, np.newaxis] * colormap(heatmap_norm)[:, :, :3]  # Use only RGB channels

        # Add to the composite image
        composite += colored_heatmap

    pass
    composite /= np.max(composite)

    # Display the composite image
    # plt.imshow(composite)
    # plt.axis('off')
    # plt.title("Composite Heatmaps")
    # plt.show()
    return composite


def visualized_fly_net_input_2():
    # input_to_cnn_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov2\saved_box_dir\box.h5"
    input_to_cnn_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints.h5"

    input_to_cnn = h5py.File(input_to_cnn_path, 'r')['/box'][70:110]
    output_from_cnn = h5py.File(input_to_cnn_path, 'r')['/confmaps'][..., 70:110].T
    frame = 1
    cam = 1

    single_output = output_from_cnn[frame, cam]
    single_image = input_to_cnn[frame, cam]
    colored_heatmaps = visualize_heatmaps(single_output)
    plt.imshow(1 - single_image[..., 1], cmap='gray')
    plt.imsave('input image example.png', 1 - single_image[..., 1], cmap='gray')
    plt.show()
    # Visualizer.visualized_channels_in_3D(single_image, type_to_display=INPUT)
    # Visualizer.visualized_channels_in_3D(single_output, type_to_display=OUTPUT)

    for cam in range(4):
        single_output = output_from_cnn[frame, cam]
        single_image = input_to_cnn[frame, cam]
        colored_heatmaps = visualize_heatmaps(single_output)
        Visualizer.visualize_channels_all_in_one(single_image, colored_heatmaps, cam+1)


def visualized_fly_net_input_output_vertical():
    input_to_cnn_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints.h5"

    input_to_cnn = h5py.File(input_to_cnn_path, 'r')['/box'][70:110]
    output_from_cnn = h5py.File(input_to_cnn_path, 'r')['/confmaps'][..., 70:110].T
    frame = 1
    cam = 1

    single_output = output_from_cnn[frame, cam]
    single_image = input_to_cnn[frame, cam]

    fig = plt.figure(figsize=(16, 8))  # Adjust the figure size
    plt.subplots_adjust(wspace=0.1)  # Minimize the space between plots

    # Input 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    Visualizer.visualized_channels_in_3D_axis(single_image, type_to_display=INPUT, ax=ax1)

    # Output 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    Visualizer.visualized_channels_in_3D_axis(single_output, type_to_display=OUTPUT, ax=ax2)

    # Display the figure
    plt.show()


def visualize_net_input_star_config():
    input_to_cnn_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints.h5"
    input_to_cnn = h5py.File(input_to_cnn_path, 'r')['/box'][70:110]
    output_from_cnn = h5py.File(input_to_cnn_path, 'r')['/confmaps'][..., 70:110].T
    frame = 1
    cam = 1
    single_output = output_from_cnn[frame, cam]
    single_image = input_to_cnn[frame, cam]
    visualize_colored_star_config(single_image)


def visualize_colored_star_config(single_image):
    # Ensure the input array is of the expected shape (192, 192, 5)
    if single_image.shape != (192, 192, 5):
        raise ValueError("Input image must have the shape (192, 192, 5)")

    # Separate the channels
    ch1, ch2, ch3, ch4, ch5 = [single_image[..., i] for i in range(5)]

    # Create blank RGB canvases for masks ch4 and ch5
    mask4_canvas = np.zeros((192, 192, 3))
    mask5_canvas = np.zeros((192, 192, 3))

    # Assign custom colors to binary masks
    mask4_canvas[..., 1] = np.clip(ch4 * 2.0, 0, 1)  # Green for Channel 4
    mask5_canvas[..., 0] = np.clip(ch5 * 2.0, 0, 1)  # Red for Channel 5
    mask5_canvas[..., 2] = np.clip(ch5 * 2.0, 0, 1)  # Add blue for Channel 5 (magenta effect)

    # Create RGB canvases for middle-left and middle-right
    ch1_canvas = np.zeros((192, 192, 3))  # Red canvas for ch1
    ch3_canvas = np.zeros((192, 192, 3))  # Blue canvas for ch3

    # Paint middle-left (ch1) as bright red
    ch1_canvas[..., 0] = np.clip(ch1 * 2.0, 0, 1)  # Assign bright red to red channel

    # Paint middle-right (ch3) as bright blue
    ch3_canvas[..., 2] = np.clip(ch3 * 3.0, 0, 1)  # Assign bright blue to blue channel

    # Create a grayscale image for the center (ch2) with contours
    center_canvas = np.stack([ch2, ch2, ch2], axis=-1)  # Convert grayscale to RGB

    # Add the perimeter of binary masks (ch4 and ch5) to center_canvas
    perimeter_ch4 = measure.find_contours(ch4, 0.5)
    perimeter_ch5 = measure.find_contours(ch5, 0.5)

    # Overlay the perimeters on the center_canvas
    for contour in perimeter_ch4:
        contour = contour.astype(int)
        center_canvas[contour[:, 0], contour[:, 1], 1] = 1  # Add green perimeter for ch4

    for contour in perimeter_ch5:
        contour = contour.astype(int)
        center_canvas[contour[:, 0], contour[:, 1], 0] = 1  # Add red perimeter for ch5
        center_canvas[contour[:, 0], contour[:, 1], 2] = 1  # Add blue perimeter for ch5 (magenta)

    # Create a figure and axes
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)  # Add spacing between subplots

    # Star configuration layout:
    # [   ] [ch4] [   ]
    # [ch1] [ch2] [ch3]
    # [   ] [ch5] [   ]

    # Add images to the corresponding subplots
    ax[0, 1].imshow(mask4_canvas)  # Top-center (Channel 4, green mask)
    ax[1, 0].imshow(ch1_canvas)  # Middle-left (Channel 1, bright red)
    ax[1, 1].imshow(center_canvas)  # Center (Channel 2, with perimeters added)
    ax[1, 2].imshow(ch3_canvas)  # Middle-right (Channel 3, bright blue)
    ax[2, 1].imshow(mask5_canvas)  # Bottom-center (Channel 5, magenta mask)

    # Remove axes for all subplots
    for row in ax:
        for a in row:
            a.axis('off')

    # Show the plot
    plt.show()


def visualize_planes():
    h5_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov3\mov3_analysis_smoothed.h5"

    points = Visualizer.get_data_from_h5(h5_path, 'points_3D')
    upper_planes = Visualizer.get_data_from_h5(h5_path, 'all_upper_planes')
    lower_planes = Visualizer.get_data_from_h5(h5_path, 'all_lower_planes')
    planes = np.concatenate((lower_planes, upper_planes), axis=1)
    Visualizer.show_points_and_wing_planes_3D(points, planes)


def run_visualize_auto_correlations():
    # path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
    # path_intact = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms"
    path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
    path_intact = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    Visualizer.visualize_auto_correlations(path_cut, path_intact)


def visualize_speeds_distribution():
    path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
    path_intact = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    Visualizer.visualize_speeds_distributions(path_cut, path_intact)


def compare_psi_to_roni():
    sagiv_movies = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    roni_results = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\manipulated_05_12_22.hdf5"
    all_left_psi = Visualizer.load_all_attributes_from_h5(sagiv_movies, attribute='wings_psi_left')
    pass


def visualize_ensemble_vs_result(movie_path):
    frames = np.arange(500, 1000)
    all_points_2D_file_list = []
    dir_path = os.path.join(movie_path)
    dirs = glob.glob(os.path.join(dir_path, "*"))
    for dir in dirs:
        if os.path.isdir(dir):
            predictions_file_h5_path = os.path.join(dir, "predicted_points_and_box.h5")
            if os.path.isfile(predictions_file_h5_path):
                points_2D = Visualizer.get_data_from_h5(predictions_file_h5_path, 'positions_pred', frames)
                all_points_2D_file_list.append(points_2D)
    box_path = glob.glob(os.path.join(movie_path, "movie*.h5"))[0]
    # box = Visualizer.get_data_from_h5(box_path, 'box', frames=frames)
    reprojecded_path = os.path.join(movie_path, "points_ensemble_smoothed_reprojected.npy")
    reprojected_points_2D = np.load(reprojecded_path)[frames]
    box = Visualizer.get_box(box_path, frames=frames)
    movie = box[..., [1, 1, 1]]

    # Assuming movie is your 5D numpy array and points is your 3D array
    def update(val):
        frame = int(slider.val)
        for i, ax in enumerate(axes.flat):
            ax.clear()
            ax.imshow(movie[frame, i])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(reprojected_points_2D[frame, i])))  # create a color array
            for model_num in range(len(all_points_2D_file_list)):
                points = all_points_2D_file_list[model_num]
                ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='.')  #
            ax.scatter(*reprojected_points_2D[frame, i].T, edgecolors=colors, facecolors='none', marker='o')
        plt.draw()

    def on_key_press(event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, 0))  # decrement slider value

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
    axes = axes.ravel()  # flatten the grid to easily iterate over it
    plt.subplots_adjust(bottom=0.2)  # make room for the slider

    slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
    slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    fig.canvas.mpl_connect('key_press_event',
                           on_key_press)  # connect the key press event to the on_key_press function

    plt.show()


def visualize_feature_points():
    from skimage import exposure

    mov = 104
    path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{mov}\saved_box_dir\box.h5"
    cam = 1
    frame = 1047
    frame = 1251
    frame = 1320
    # end = 1047
    # Load data
    with h5py.File(path, "r") as h5_file:
        box = 1 - h5_file["/array"][frame][cam, :, :, 1]

    # Apply histogram equalization to non-1 pixels
    # non_1_mask = box < 1
    # box = box.copy()
    # box[non_1_mask] = exposure.equalize_hist(box[non_1_mask])

    points_2D = np.load(
        rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{mov}\points_ensemble_smoothed_reprojected.npy"
    )
    all_points_inds = np.arange(18)
    all_points_inds = all_points_inds[~np.isin(all_points_inds, [7, 15])]
    points_2D = points_2D[frame, cam, all_points_inds, :]

    # Define point groups
    points_left_wing = np.arange(0, 7)  # Left-wing points
    points_right_wing = np.arange(7, 14)  # Right-wing points
    points_tail = np.arange(14, 16)  # Tail points

    # Define a color palette
    color_palette = ["blue", "orange", "green", "purple", "cyan", "magenta", "yellow"]

    # Assign colors based on index modulo the palette size
    left_wing_colors = [color_palette[i % len(color_palette)] for i in points_left_wing]
    right_wing_colors = [color_palette[i % len(color_palette)] for i in points_right_wing]
    tail_color = "red"

    # Plot setup
    plt.figure(figsize=(10, 10))
    plt.imshow(box, cmap="gray")
    plt.title("Feature Points Visualization with Histogram Equalization", fontsize=16)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)

    # Plot left-wing points
    for i, color in zip(points_left_wing, left_wing_colors):
        plt.scatter(points_2D[i, 0], points_2D[i, 1], color=color, label=f"Left Wing {i}")

    # Plot right-wing points
    for i, color in zip(points_right_wing, right_wing_colors):
        plt.scatter(points_2D[i, 0], points_2D[i, 1], color=color, label=f"Right Wing {i}")

    # Plot tail points
    plt.scatter(
        points_2D[points_tail, 0],
        points_2D[points_tail, 1],
        color=tail_color,
        label="Tail (14-15)"
    )

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc="upper right")

    plt.show()


if __name__ == '__main__':
    # Visualizer.input_to_output()
    # visualized_fly_net_input_2()
    visualized_fly_net_input_output_vertical()
    # all_sampled_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\all_flies.npy"
    # all_flies = np.load(all_sampled_path)
    # Visualizer.visualize_monte_carlo_3d(all_flies)



    # h5_path_movie_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\mov78_analysis_smoothed.h5"
    # reprojected_points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\points_ensemble_smoothed_reprojected.npy"
    # box_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\movie_78_10_4868_ds_3tc_7tj.h5"
    # Visualizer.create_movie_mp4(h5_path_movie_path, mode=DISPLAY, save_frames=None,
    #                  reprojected_points_path=reprojected_points_path, box_path=box_path,
    #                  save_path="movie_gif.gif", rotate=False)
    # run_visualize_auto_correlations()

    # visualize_speeds_distribution()

    # visualize_feature_points()
    # movie_path = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight\mov1"
    # visualize_ensemble_vs_result(movie_path)

    #
    # # movie_dir_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov8"
    # # Visualizer.visualize_3D_point_selection(movie_dir_path)
    # # visualize_net_input_star_config()
    # # visualize_planes()
    # # path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
    # # path_intact = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms"
    # # Visualizer.visualize_auto_correlations(path_cut, path_intact)
    # # Visualizer.visualize_center_mass_speed_distributions(path_cut, path_intact)
    # # visualized_fly_net_input()
    # # path_h5 = r"C:\Users\amita\OneDrive\Desktop\temp\movies\mov10\movie_10_300_3008_ds_3tc_7tj.h5"
    # # Visualizer.display_movie_from_path(path_h5)
    #
    # # directory_path = fr"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
    # # traverse_and_plot(directory_path)
    # h5_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov8\mov8_analysis_smoothed.h5"
    # # Visualizer.visualize_analysis_3D_html(h5_path, filename="3d_analysis_with_slider.html")
    #
    # # base_path = "free 24-1 movies"
    # # base_path = "dark 24-1 movies"
    # # base_path = "example datasets"
    # # base_path = "roni dark 60ms"
    # # create_mp4_directory(base_path)
    #
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies\mov14\mov14_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies\mov24\mov24_analysis_smoothed.h5"
    # # h5_path = r"C:\Users\amita\OneDrive\Desktop\temp\mov24_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov24\mov24_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov3\mov3_analysis_smoothed.h5"
    # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov53\mov53_analysis_smoothed.h5"
    # # Visualizer.visualize_analysis_3D_html(h5_path)
    # # Visualizer.visualize_analisys_3D(h5_path, ACTION=DISPLAY)
    #
    # # movie_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\arranged movies\mov62\movie_62_160_1888_ds_3tc_7tj.h5"
    # # Visualizer.display_movie_from_path(movie_path)
    #
    # # display analysis
    # # movie = 101
    # # h5_path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{movie}\mov{movie}_analysis_smoothed.h5"
    # # reprojected_points_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\points_ensemble_smoothed_reprojected.npy"
    # # box_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\movie_78_10_4868_ds_3tc_7tj.h5"
    #
    # # base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies\mov6"
    # # box_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies\mov6\movie_6_100_498_ds_3tc_7tj.h5"
    # # mov_num = 6
    # # create_gif_one_movie(base_path=base_path,
    # #                      mov_num=mov_num,
    # #                      box_path=box_path)
    #
    # # display analysis
    # # reprojected_points_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\arranged movies\mov53\points_ensemble_reprojected.npy"
    # # box_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\arranged movies\mov53\movie_53_10_2398_ds_3tc_7tj.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\arranged movies\mov53\mov53_analysis_smoothed.h5"
    # # Visualizer.create_gif_for_movie(h5_path, reprojected_points_path=reprojected_points_path, box_path=box_path,
    # #                                 save_path='analysis_reprojected_unsmoothed.gif')
    #
    # # mov_num = 20
    # # h5_path = fr"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov63\mov63_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov35\mov35_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov46\mov46_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov77\mov77_analysis_smoothed.h5"
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov3\mov3_analysis_smoothed.h5"
    #
    # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov51 problem\mov51 problem_analysis_smoothed.h5"
    # Visualizer.visualize_analisys_3D(h5_path)
    #
    # # Visualizer.plot_all_body_data(h5_path)
    # # Visualizer.visualize_analisys_3D(h5_path)
    # # Visualizer.display_omega_body(h5_path)
    # # Visualizer.plot_psi_or_theta_vs_phi(h5_path, wing='right')
    # # Visualizer.plot_psi_or_theta_vs_phi(h5_path, wing='left')
    # # Visualizer.show_left_vs_right_amplitude(h5_path)
    # # Visualizer.compare_body_angles(h5_path)
    # # Visualizer.plot_theta_vs_phi(h5_path, wing_bits=[1,2,3,4,5,6,7])
    # # Visualizer.show_amplitude_graph(h5_path)
    # # Visualizer.show_left_vs_right_amplitude(h5_path)
    #
    # # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\mov53\mov53_analysis_smoothed.h5"
    # # Visualizer.visualize_analisys_3D(h5_path)
    # # display 3D points
    # # path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov35\saved_box_dir\box.h5"
    # # box = h5py.File(path, "r")["/array"][..., [1, 3, 4]]
    # # Visualizer.display_movie_from_box(box)
    #
    # # points_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov35\movie_35_130_1478_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Jun 19\points_3D.npy"
    # # points_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov35\movie_35_130_1478_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Jun 19_01\points_3D.npy"
    # # points_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov35\movie_35_130_1478_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Jun 19_06\points_3D.npy"
    #
    # # points_path = r"C:\Users\amita\OneDrive\Desktop\points_3D_ensemble_best_method.npy"
    # # points_3D = np.load(points_path)
    # # Visualizer.show_points_in_3D(points_3D)
    #
    # display box and 2D predictions
    mov = 104
    path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{mov}\saved_box_dir\box.h5"
    start = 1000
    end = 1100
    box = h5py.File(path, "r")["/array"][start:end]
    # points_2D = np.load(rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{mov}\points_ensemble_smoothed_reprojected.npy")
    path_h5 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov104\movie_104_10_5048_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_May 02\predicted_points_and_box.h5"
    points_2D = h5py.File(path_h5, 'r')['/positions_pred'][:]
    # Visualizer.show_predictions_all_cams(box, points_2D[start:end], scatter=False)


    box_path = fr"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\movie_78_10_4868_ds_3tc_7tj.h5"
    h5_path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\mov78_analysis_smoothed.h5"
    reprojected_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov78\points_ensemble_smoothed_reprojected.npy"

    # box_path = rf"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov53\movie_53_10_2398_ds_3tc_7tj.h5"
    # h5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov53\mov53_analysis_smoothed.h5"
    # reprojected_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies\mov53\points_ensemble_smoothed_reprojected.npy"

    # box_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov1\movie_1_10_4410_ds_3tc_7tj.h5"
    # h5_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov1\mov1_analysis_smoothed.h5"
    # reprojected_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov1\points_ensemble_smoothed_reprojected.npy"


    # self occluding wings
    # box_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov10\movie_10_300_3008_ds_3tc_7tj.h5"
    # h5_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov10\mov10_analysis_smoothed.h5"
    # reprojected_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\mov10\points_ensemble_smoothed_reprojected.npy"

    # Visualizer.visualize_analisys_3D(h5_path, DISPLAY)
    Visualizer.visualize_points_and_images(h5_path=h5_path,
                                           box_path=box_path,
                                           reprojected_points_path=reprojected_path)

    # # display box and 2D predictions
    # # predicted_box_path = r"C:\Users\amita\OneDrive\Desktop\temp\predicted_points_and_box.h5"
    # # box_path = r"C:\Users\amita\OneDrive\Desktop\temp\box.h5"
    # # box = h5py.File(box_path, "r")["/array"][:500]
    # # points_2D = h5py.File(predicted_box_path, "r")["/positions_pred"][:500]
    # # Visualizer.show_predictions_all_cams(box, points_2D)

    # # visualize model selection
    # # path = r"C:\Users\amita\OneDrive\Desktop\temp\all_models_combinations.npy"
    # # # path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example datasets\mov53\all_models_combinations.npy"
    # # path = r"C:\Users\amita\OneDrive\Desktop\temp\all_models_combinations.npy"
    # # all_models_combinations = np.load(path)
    # # Visualizer.visualize_models_selection(all_models_combinations)
