import itertools

import cv2
import h5py
import numpy as np

# distortion_coeffs = np.array([[-1.3083, 135.2352, 0, 0, 0],
#                      [-0.5633, 22.7305, 0, 0, 0],
#                      [-0.3261, 17.9664, 0, 0, 0],
#                      [-0.2860, 17.9070, 0, 0, 0]], dtype=np.float32)

class Triangulate:
    def __init__(self, config):
        self.config = config
        self.calibration_data_path = self.config["calibration data path"]
        print("Calibration data path:", self.calibration_data_path)
        self.image_hight = self.config["IMAGE_HEIGHT"]
        self.image_width = self.config["IMAGE_WIDTH"]
        self.rotation_matrix = self.load_rotation_matrix()
        self.camera_matrices = self.load_camera_matrices()
        self.inv_camera_matrices = self.load_inv_camera_matrices()
        self.camera_centers = self.load_camera_centers()
        self.Ks = self.load_Ks()
        self.Rs = self.load_Rs()
        self.translations = self.load_translations()
        self.num_cameras = len(self.camera_centers)
        self.all_couples = Triangulate.get_all_combinations(n=3)
        self.all_cameras_combinations = Triangulate.get_all_combinations(n=5)
        # self.distortion = False
        # try:
        #     self.distortion_params = self.load_distortion_params()
        #     self.distortion = True
        # except:
        #     print(f"no distortion parameters")

    def load_Ks(self):
        Ks = h5py.File(self.calibration_data_path, "r")["K_matrices"][:].T
        for i, K in enumerate(Ks):
            K = K / K[2, 2]
            Ks[i] = K
        return Ks

    def load_distortion_params(self):
        distortion_params_full = np.zeros((4, 5))
        distortion_params = h5py.File(self.calibration_data_path, "r")["distortion_params"][:].T
        num_params_per_cam = distortion_params.shape[1]
        distortion_params_full[:, :num_params_per_cam] = distortion_params
        return distortion_params_full
    def load_Rs(self):
        return h5py.File(self.calibration_data_path, "r")["rotation_matrices"][:].T

    def load_translations(self):
        return h5py.File(self.calibration_data_path, "r")["translations"][:].T

    def load_rotation_matrix(self):
        return h5py.File(self.calibration_data_path, "r")["rotation_matrix"][:].T

    def load_camera_matrices(self):
        return h5py.File(self.calibration_data_path, "r")["camera_matrices"][:].T

    def load_inv_camera_matrices(self):
        return h5py.File(self.calibration_data_path, "r")["inv_camera_matrices"][:].T

    def load_camera_centers(self):
        return h5py.File(self.calibration_data_path, "r")["camera_centers"][:].T

    def triangulate_2D_to_3D_svd(self, points_2D, cropzone):
        num_frames, _, num_joints, _ = points_2D.shape
        points_3D_all = np.zeros((num_frames, num_joints, 6, 3))
        reprojection_errors = np.zeros((num_frames, num_joints, 6))
        traingulation_errors = np.zeros((num_frames, num_joints, 6))
        for i, cameras_couple in enumerate(self.all_couples):
            for joint in range(num_joints):
                points_2D_sub_views = points_2D[:, cameras_couple, joint, :]
                points_2D_sub_views = np.transpose(points_2D_sub_views, [1, 0, 2])
                # projection_matrices_sub_views = self.camera_matrices[cameras_couple, ...]
                cropzone_sub_view = cropzone[:, cameras_couple, :]
                points_3D_sub, reprojection_errors_j_i, traingulation_errors_j_i = self.triangulate_points_svd(points_2D_sub_views, cropzone_sub_view, cameras_couple)
                points_3D_all[:, joint, i, :] = points_3D_sub
                reprojection_errors[:, joint, i] = reprojection_errors_j_i
                traingulation_errors[:, joint, i] = traingulation_errors_j_i
        return points_3D_all, reprojection_errors, traingulation_errors

    def triangulate_points_svd(self, points_2d, cropzone, cameras_couple):
        """
        Args:
            points_2d (array like): Sequence of 2D points per view.
                If there are N points and M views, points_2d can an array of (M, N, 2) or a list of (N, 2) arrays.
            projection_matrices (array like): Sequence of projection matrices.
                If there are M views, projection_matrices can an array of (M, 3, 4) or a list of (3, 4) arrays.
        Returns:
            ndarray: Computed 3d points. Array of shape (n x 3)
            reprojection error for each point. the avarege

        """
        points_2d = np.asarray(points_2d)
        projection_matrices = self.camera_matrices[cameras_couple, ...]
        assert (points_2d.shape[0] == projection_matrices.shape[0])
        n_views = points_2d.shape[0]
        n_points = points_2d.shape[1]

        points_2d_h = self.uncrop(points_2d, cropzone)

        A = np.zeros((3 * n_views, (4 + n_views)), dtype=np.float64)
        points_3d = []

        for i in range(n_points):
            for j in range(n_views):
                A[(j * 3):3 + (j * 3), :4] = -projection_matrices[j]
                A[(j * 3):3 + (j * 3), 4 + j] = points_2d_h[j, i, :]

            u, s, vh = np.linalg.svd(A)
            X = vh[-1, :][:4]  # right vector corresponding to the smallest singular value
            X = Triangulate.h2e_coords(X)
            points_3d.append(X)

        points_3d = np.array(points_3d)

        # calculate the reprojection error
        reprojection_errors = self.extract_reprojection_error(points_2d_h, points_3d, projection_matrices)
        traingulation_errors = self.extract_triangulation_error(points_2d_h, points_3d, cameras_couple)

        points_3d = (self.rotation_matrix @ points_3d.T).T
        return points_3d, reprojection_errors, traingulation_errors

    def uncrop(self, points_2d, cropzone, add_one=True):
        num_cams, num_frames, _ = points_2d.shape
        new_shape = list(points_2d.shape)
        if add_one:
            new_shape[-1] += 1
        points_2d_h = np.zeros(new_shape)
        for frame in range(num_frames):
            for cam in range(num_cams):
                x = cropzone[frame, cam, 1] + points_2d[cam, frame, 0]
                y = cropzone[frame, cam, 0] + points_2d[cam, frame, 1]
                # do undistort
                # if self.distortion:
                #     x, y = self.undistort_point(cam, x, y)
                y = self.image_hight + 1 - y
                if add_one:
                    point = [x, y, 1]
                else:
                    point = [x, y]
                points_2d_h[cam, frame, :] = point
        return points_2d_h

    def undistort_point(self, cam, x, y):
        pixel_coords = np.array([[[x, y]]], dtype=np.float32)
        K = self.Ks[cam]
        dist_coeffs = self.distortion_params[cam]
        point = cv2.undistortPoints(pixel_coords, K, distCoeffs=dist_coeffs, P=K)
        x, y = point[0, 0, :]
        return x, y

    @staticmethod
    def h2e_coords(points_h):
        """
        Converts points from homogeneous to euclidean coordinates. For example, in 2D euclidean space: (x,y,z)->(x/z, y/z)
        Args:
            points (array like): Array like of N+1 dimensional points. The array can have arbitrary shape as long as the last dimension is N+1.
        Returns:
            array: Array of N dimensional points. The returned array has the shape as the input points except for (N-1)->N in the last dimension.

        """
        points = points_h[..., :-1] / points_h[..., -1:]
        return points

    def triangulate_2D_to_3D_rays_optimization(self, points_2D, cropzone):
        num_frames, _, num_points, _ = points_2D.shape
        points_2D_uncropped = self.get_uncropped_xy1(points_2D, cropzone)
        all_points_3D = np.zeros((num_frames, num_points, 6, 3))
        triangulation_errors = np.zeros((num_frames, num_points, 6))
        reprojection_errors = np.zeros((num_frames, num_points, 6))
        for point in range(num_points):
            for i, couple in enumerate(self.all_couples):
                for frame in range(num_frames):
                    cam_a, cam_b = couple

                    inv_cam_mat_a = self.inv_camera_matrices[cam_a]  # get the inverse camera matrix
                    center_a = self.camera_centers[cam_a]  # camera center
                    pnt_a_2d = points_2D_uncropped[frame, cam_a, point, :]  # get point
                    pnt_a_projected = inv_cam_mat_a @ pnt_a_2d  # get projection
                    pnt_a_projected = pnt_a_projected[:-1] / pnt_a_projected[-1]  # normalize projection

                    inv_cam_mat_b = self.inv_camera_matrices[cam_b]
                    pnt_b_2d = points_2D_uncropped[frame, cam_b, point, :]
                    center_b = self.camera_centers[cam_b]
                    pnt_b_projected = inv_cam_mat_b @ pnt_b_2d
                    pnt_b_projected = pnt_b_projected[:-1] / pnt_b_projected[-1]

                    centers = np.vstack((center_a, center_b))
                    projected_pnts = np.vstack((pnt_a_projected, pnt_b_projected))

                    point_3d, triangulation_error = np.squeeze(Triangulate.lineIntersect3D(centers, projected_pnts))
                    point_3d = self.rotation_matrix @ point_3d
                    all_points_3D[frame, point, i, :] = point_3d
                    triangulation_errors[frame, point, i] = triangulation_error

                # get the reprojection errors
                # points_2d_h, points_3d, projection_matrices
                points_2d_h = np.transpose(points_2D_uncropped[:, couple, point, :], [1, 0, 2])
                points_3d = all_points_3D[:, point, i, :]
                points_3d_unrotated = (self.rotation_matrix.T @ points_3d.T).T
                projection_matrices = self.camera_matrices[couple, :]
                reprojection_errors_p_i = self.extract_reprojection_error(points_2d_h, points_3d_unrotated, projection_matrices)
                reprojection_errors[:, point, i] = reprojection_errors_p_i
        return all_points_3D, reprojection_errors, triangulation_errors

    def rays_midpoint_traingulation(self, points_2D, cropzone):
        """
        this function perfomrs triangulation from 2 views, by using Cyrill Stachniss lecture notes
        from the video:
        Triangulation for Image Pairs (Cyrill Stachniss)
        https://www.youtube.com/watch?v=UZlRhEUWSas&t=51s
        """
        num_frames, _, num_points, _ = points_2D.shape
        all_points_3D = np.zeros((num_frames, num_points, 6, 3))
        triangulation_errors = np.zeros((num_frames, num_points, 6))
        reprojection_errors = np.zeros((num_frames, num_points, 6))
        points_2d_h = self.get_uncropped_xy1(points_2D, cropzone)
        for frame in range(num_frames):
            for i, couple in enumerate(self.all_couples):
                for point_ind in range(num_points):
                    rays = []
                    camera_centers = []
                    for cam in couple:
                        K = self.Ks[cam]
                        R = self.Rs[cam]
                        K_inv = np.linalg.inv(K)
                        point_pixel = points_2d_h[frame, cam, point_ind, :]
                        Xk = np.dot(K_inv, point_pixel)
                        ray_vec = np.dot(R.T, Xk)
                        ray_vec /= np.linalg.norm(ray_vec)
                        camera_center = self.camera_centers[cam]
                        rays.append(ray_vec)
                        camera_centers.append(camera_center)

                    # construct Ax = b
                    Xo_1, r = camera_centers[0], rays[0]
                    Xo_2, s = camera_centers[1], rays[1]
                    A = np.zeros((2, 2))
                    b = np.zeros((2, 1))
                    A[0, 0] = r.T @ r
                    A[0, 1] = -s.T @ r
                    A[1, 0] = r @ s
                    A[1, 1] = -s.T @ s
                    b[0] = (Xo_2 - Xo_1).T @ r
                    b[1] = (Xo_2 - Xo_1).T @ s

                    # solve Ax = b and get lamda and mu
                    lam, mu = np.linalg.solve(A, b)

                    # get the point by applying lambda and mu
                    p1 = Xo_1 + lam * r
                    p2 = Xo_2 + mu * s
                    p = (p1 + p2)/2

                    # get triangulation error
                    d = np.linalg.norm(p1 - p2) / 2

                    all_points_3D[frame, point_ind, i, :] = self.rotation_matrix @ p
                    triangulation_errors[frame, point_ind, i] = d

                    # get reprojection errors
                    rep_errors = np.zeros(2)
                    for j, cam in enumerate(couple):
                        p_reprojected = self.camera_matrices[cam] @ np.append(p, 1)
                        p_reprojected = p_reprojected / p_reprojected[-1]
                        p_2d_orig = points_2d_h[frame, cam, point_ind]
                        reprojection_error = np.linalg.norm(p_2d_orig - p_reprojected)
                        rep_errors[j] = reprojection_error
                    rep_error = rep_errors.mean()
                    reprojection_errors[frame, point_ind, i] = rep_error
        return points_3D_all, reprojection_errors, triangulation_errors
    def get_uncropped_xy1(self, points_2D, cropzone):
        new_shape = list(points_2D.shape)
        new_shape[-1] += 1
        points_2D_uncropped = np.zeros(new_shape)
        num_frames, num_cams, num_points, _ = new_shape
        for frame in range(num_frames):
            for cam in range(num_cams):
                for pnt in range(num_points):
                    x = cropzone[frame, cam, 1] + points_2D[frame, cam, pnt, 0]
                    y = cropzone[frame, cam, 0] + points_2D[frame, cam, pnt, 1]
                    y = self.image_hight + 1 - y
                    point = [x, y, 1]
                    points_2D_uncropped[frame, cam, pnt, :] = point
        return points_2D_uncropped

    def get_reprojections(self, points_3D, cropzone):
        """
        this function get's an array of size (num_frames, num_joints, 3) and returns the reprojected points
        to each of the cameras, meaning an array of (num_frames, num_cams, num_joints, 2)
        Args:
            points_3D: (num_frames, num_joints, 3)

        Returns:
            points_2D_reprojected: (num_frames, num_cams, num_joints, 2)
        """
        num_frames, num_joints, _ = points_3D.shape
        points_2D_reprojected = np.zeros((num_frames, self.num_cameras, num_joints, 2))
        for frame in range(num_frames):
            for joint in range(num_joints):
                point = points_3D[frame, joint, :]
                point_rot = self.rotation_matrix.T @ point
                point_rot_h = np.append(point_rot, 1)
                for cam in range(self.num_cameras):
                    y_crop, x_crop = cropzone[frame, cam, :]
                    camera_mat = self.camera_matrices[cam, ...]
                    point_2D_hom = camera_mat @ point_rot_h
                    point_2D = point_2D_hom[:-1] / point_2D_hom[-1]
                    xp, yp = point_2D
                    yp = self.image_hight + 1 - yp
                    x = xp - x_crop
                    y = yp - y_crop
                    point_2D_cropped = np.array([x, y])
                    points_2D_reprojected[frame, cam, joint, :] = point_2D_cropped


                    # experiment
                    # K = self.Ks[cam]
                    # dx = x_crop
                    # dy = 800 + 1 - y_crop - 192
                    # K_prime = K.copy()
                    # K_prime[0, 2] -= dx  # adjust x-coordinate of the principal point
                    # K_prime[1, 2] -= dy  # adjust y-coordinate of the principal point
                    # R = self.Rs[cam]
                    # t = self.translations[cam]
                    # # M = K @ np.column_stack((R, t))
                    # # M /= M[-1,-1]
                    # M_prime = K_prime @ np.column_stack((R, t))
                    # M_prime /= M_prime[-1, -1]
                    # p2d_h = M_prime @ point_rot_h
                    # p2d = p2d_h[:-1] / p2d_h[-1]
                    # p2d[1] = 192 - p2d[1]
                    # print(f"{np.mean(np.abs(p2d - point_2D_cropped))}")

        # self.estimating_camera_experiment(num_frames, points_2D_reprojected, points_3D)
        return points_2D_reprojected

    def estimating_camera_experiment(self, num_frames, points_2D_reprojected, points_3D):
        arr = np.arange(num_frames)
        N = 10
        chosen_inds = np.random.choice(arr, N, replace=False)
        pts_3D = np.squeeze(points_3D[chosen_inds, :])
        pts_2D_reprojected = np.squeeze(points_2D_reprojected[chosen_inds, :])
        camera_matrices = []
        for cam in range(self.num_cameras):
            points_2D = pts_2D_reprojected[:, cam, :]
            P = self.estimate_projection_matrix_dlt(pts_3D, points_2D)
            camera_matrices.append(P)
        cam1 = 1
        cam2 = 3
        P1 = camera_matrices[cam1]
        P2 = camera_matrices[cam2]
        X1 = pts_2D_reprojected[:, cam1, :]
        X2 = pts_2D_reprojected[:, cam2, :]
        points_3d = cv2.triangulatePoints(P1, P2, X1.T, X2.T).T
        points_3d = points_3d[:, :-1] / points_3d[:, -1:]
        error = np.mean(np.abs(points_3d - pts_3D))

    @staticmethod
    def estimate_projection_matrix_dlt(points_3d, points_2d):
        assert len(points_2d) == len(points_3d)
        assert len(points_2d) >= 6

        A = []

        for i in range(len(points_2d)):
            X, Y, Z = points_3d[i]
            x, y = points_2d[i]
            A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
            A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

        _, _, V = np.linalg.svd(A)
        P = V[-1].reshape(3, 4)
        P /= P[-1, -1]
        return P

    @staticmethod
    def lineIntersect3D(centers, projected_points):
        # Assuming PA and PB are numpy arrays

        # N lines described as vectors
        rays = projected_points - centers

        # Normalize vectors
        ni = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        nx, ny, nz = ni.T

        # Calculate S matrix
        SXX = np.sum(nx ** 2 - 1)
        SYY = np.sum(ny ** 2 - 1)
        SZZ = np.sum(nz ** 2 - 1)
        SXY = np.sum(nx * ny)
        SXZ = np.sum(nx * nz)
        SYZ = np.sum(ny * nz)
        S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])

        # Calculate C vector
        CX = np.sum(centers[:, 0] * (nx ** 2 - 1) + centers[:, 1] * (nx * ny) + centers[:, 2] * (nx * nz))
        CY = np.sum(centers[:, 0] * (nx * ny) + centers[:, 1] * (ny ** 2 - 1) + centers[:, 2] * (ny * nz))
        CZ = np.sum(centers[:, 0] * (nx * nz) + centers[:, 1] * (ny * nz) + centers[:, 2] * (nz ** 2 - 1))
        C = np.array([CX, CY, CZ])

        # Solve for intersection point
        P_intersect = np.linalg.solve(S, C)

        # Calculate distance from intersection point to each line
        N = 2
        distances = np.zeros(N, )
        for i in range(2):
            ui = np.dot(P_intersect - centers[i, :], rays[i, :]) / np.dot(rays[i, :], rays[i, :])
            distances[i] = np.linalg.norm(P_intersect - centers[i, :] - ui * rays[i, :])

        return P_intersect, np.mean(distances)

    ###
    def triangulate_2D_to_3D_reprojection_optimization(self, points_2D, cropzone):
        num_frames, _, num_joints, _ = points_2D.shape
        points_3D_all = np.zeros((num_frames, num_joints, 6, 3))
        reprojection_errors = np.zeros((num_frames, num_joints, 6))
        triangulation_errors = np.zeros((num_frames, num_joints, 6))
        for i, sub in enumerate(self.all_couples):
            for joint in range(num_joints):
                points_2D_sub_views = points_2D[:, sub, joint, :]
                points_2D_sub_views = np.transpose(points_2D_sub_views, [1, 0, 2])
                cropzone_sub_view = cropzone[:, sub, :]
                points_3D_sub, reprojection_errors_j_i, triangulation_errors_j_i = self.triangulate_points_cv2(points_2D_sub_views, cropzone_sub_view, sub)
                points_3D_all[:, joint, i, :] = points_3D_sub
                reprojection_errors[:, joint, i] = reprojection_errors_j_i
                triangulation_errors[:, joint, i] = triangulation_errors_j_i
        return points_3D_all, reprojection_errors, triangulation_errors

    def triangulate_points_all_possible_views(self, points_2D, cropzone):
        num_frames, _, num_joints, _ = points_2D.shape
        points_3D_all = np.zeros((num_frames, num_joints, len(self.all_cameras_combinations), 3))
        reprojection_errors = np.zeros((num_frames, num_joints, len(self.all_cameras_combinations)))
        for i, sub in enumerate(self.all_cameras_combinations):
            for joint in range(num_joints):
                points_2D_sub_views = points_2D[:, sub, joint, :]
                points_2D_sub_views = np.transpose(points_2D_sub_views, [1, 0, 2])
                cropzone_sub_view = cropzone[:, sub, :]
                points_3D_sub, reprojection_errors_j_i = self.triangulate_points_multiviews(
                    points_2D_sub_views, cropzone_sub_view, sub)
                points_3D_all[:, joint, i, :] = points_3D_sub
                reprojection_errors[:, joint, i] = reprojection_errors_j_i
        return points_3D_all, reprojection_errors

    def triangulate_points_multiviews(self, points_2D_sub_views, cropzone_sub_views, sub):
        projection_matrices = self.camera_matrices[sub, ...]
        points_2d = self.uncrop(points_2D_sub_views, cropzone_sub_views, add_one=False)
        points_2d_h = self.uncrop(points_2D_sub_views, cropzone_sub_views, add_one=True)
        points_3D = Triangulate.multiview_triangulation(camera_matrices=projection_matrices, points_2d_list=points_2d)
        reprojection_errors = self.extract_reprojection_error(points_2d_h, points_3D, projection_matrices)
        points_3D = (self.rotation_matrix @ points_3D.T).T
        return points_3D, reprojection_errors

    def triangulate_points_cv2(self, points_2d, cropzone, sub):
        """

        """
        points_2d = np.asarray(points_2d)
        projection_matrices = self.camera_matrices[sub, ...]
        assert (points_2d.shape[0] == projection_matrices.shape[0])

        P1, P2 = projection_matrices
        points_2d_h = self.uncrop(points_2d, cropzone)
        X1, X2 = self.uncrop(points_2d, cropzone)
        X1, X2 = X1[:, :2], X2[:, :2]
        points_3d = cv2.triangulatePoints(P1, P2, X1.T, X2.T).T
        points_3d = points_3d[:, :-1] / points_3d[:, -1:]
        # calculate the reprojection error
        reprojection_errors = self.extract_reprojection_error(points_2d_h, points_3d, projection_matrices)
        traingulation_errors = self.extract_triangulation_error(points_2d_h, points_3d, sub)
        points_3d = (self.rotation_matrix @ points_3d.T).T
        return points_3d, reprojection_errors, traingulation_errors

    ###
    def extract_triangulation_error(self, points_2d_h, points_3d, sub):
        """

        Args:
            points_2d_h: an array of size (N, 3) of 2D humogenious points
            points_3d: an array of (N,3) of 3D points
            sub: a tuple of size 2 of the 2 chosen cameras for triangulation

        Returns:
            triangulation_error: an array of size N of triangulation errors
        """
        num_frames = points_2d_h.shape[1]
        cam_a, cam_b = sub
        triangulation_error = np.zeros((num_frames, 2))
        for frame in range(num_frames):
            inv_cam_mat_a = self.inv_camera_matrices[cam_a]  # get the inverse camera matrix
            center_a = self.camera_centers[cam_a]  # camera center
            pnt_a_2d = points_2d_h[0, frame, :]  # get point
            pnt_a_projected = inv_cam_mat_a @ pnt_a_2d  # get projection
            pnt_a_projected = pnt_a_projected[:-1] / pnt_a_projected[-1]  # normalize projection

            inv_cam_mat_b = self.inv_camera_matrices[cam_b]
            pnt_b_2d = points_2d_h[1, frame, :]
            center_b = self.camera_centers[cam_b]
            pnt_b_projected = inv_cam_mat_b @ pnt_b_2d
            pnt_b_projected = pnt_b_projected[:-1] / pnt_b_projected[-1]

            centers = np.vstack((center_a, center_b))
            projected_points = np.vstack((pnt_a_projected, pnt_b_projected))
            rays = projected_points - centers

            P_intersect = points_3d[frame, :]
            for i in range(2):
                ui = np.dot(P_intersect - centers[i, :], rays[i, :]) / np.dot(rays[i, :], rays[i, :])
                distance = np.linalg.norm(P_intersect - centers[i, :] - ui * rays[i, :])
                triangulation_error[frame, i] = distance
        triangulation_error = np.mean(triangulation_error, axis=-1)
        return triangulation_error

    @staticmethod
    def extract_reprojection_error(points_2d_h, points_3d, projection_matrices):
        reprojection_errors = np.zeros((points_3d.shape[0], len(points_2d_h)))
        points_3d_hom = np.insert(points_3d, 3, np.ones((points_3d.shape[0])), axis=1)
        for i, proj_mat in enumerate(projection_matrices):
            projected_2D_i = (proj_mat @ points_3d_hom.T).T
            projected_2D_i = Triangulate.h2e_coords(projected_2D_i)
            errors_i = np.linalg.norm(projected_2D_i - points_2d_h[i, :, :-1], axis=1)
            reprojection_errors[:, i] = errors_i
        reprojection_errors = np.mean(reprojection_errors, axis=-1)
        return reprojection_errors

    @staticmethod
    def custom_linear_triangulation(Pa, Pb, points_a, points_b):
        N = points_a.shape[0]
        A = np.zeros((N, 4, 4))
        p1a, p2a, p3a = Pa[0, :], Pa[1, :], Pa[2, :]
        p1b, p2b, p3b = Pb[0, :], Pb[1, :], Pb[2, :]
        A[:, 0] = points_a[:, 0].reshape(-1, 1) * p3a - p1a
        A[:, 1] = points_a[:, 1].reshape(-1, 1) * p3a - p2a
        A[:, 2] = points_b[:, 0].reshape(-1, 1) * p3b - p1b
        A[:, 3] = points_b[:, 1].reshape(-1, 1) * p3b - p2b

        # Perform SVD on each A matrix
        _, _, Vt = np.linalg.svd(A)
        # take last row of Vt that corresponds to last column of V
        X = Vt[:, -1, :]
        X = X[:, :-1] / X[:, -1:]
        return X

    @staticmethod
    def multiview_triangulation(camera_matrices, points_2d_list):
        """
        Perform linear triangulation from multiple views.

        Args:
        - camera_matrices (list of np.ndarray): List of camera projection matrices (each of shape (3, 4)).
        - points_2d_list (list of np.ndarray): List of arrays of 2D points (each array of shape (N, 2)),
                                                where N is the number of points.

        Returns:
        - X (np.ndarray): Triangulated 3D points (of shape (N, 3)).
        """
        assert len(camera_matrices) == len(
            points_2d_list), "Number of camera matrices must equal number of 2D points lists"

        num_views = len(camera_matrices)
        num_points = points_2d_list[0].shape[0]

        # Initialize A matrix
        A = np.zeros((num_points, 2 * num_views, 4))

        for i, (P, points_2d) in enumerate(zip(camera_matrices, points_2d_list)):
            p1, p2, p3 = P[0, :], P[1, :], P[2, :]
            A[:, 2 * i] = points_2d[:, 0].reshape(-1, 1) * p3 - p1
            A[:, 2 * i + 1] = points_2d[:, 1].reshape(-1, 1) * p3 - p2

        # Perform SVD on each A matrix
        _, _, Vt = np.linalg.svd(A)

        # Take the last row of Vt that corresponds to the last column of V
        X = Vt[:, -1, :]

        # Normalize the points
        X = X[:, :-1] / X[:, -1:]

        return X

    @staticmethod
    def get_all_combinations(n=3):
        s = {0, 1, 2, 3}
        all_subs = []
        for i in range(2, n):
            subs = Triangulate.findsubsets(s, i)
            all_subs += subs
        return all_subs

    @staticmethod
    def findsubsets(s, n):
        return list(itertools.combinations(s, n))


if __name__ == '__main__':
    import json
    configuration_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\2D_to_3D_config.json"
    point_2D_path = r"C:\Users\amita\OneDrive\Desktop\temp\predicted_points_and_box.h5"
    box_path = r"C:\Users\amita\OneDrive\Desktop\temp\movie_35_250_1108_ds_3tc_7tj.h5"
    with open(configuration_path) as C:
        config = json.load(C)
        tr = Triangulate(config)
        points_2D = h5py.File(point_2D_path, 'r')['/positions_pred'][:100]

        left_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        right_indices = [8, 9, 10, 11, 12, 13, 14, 15]

        # cam = 1
        # left = points_2D[:, cam, left_indices, :]
        # right = points_2D[:, cam, right_indices, :]
        # points_2D[:, cam, left_indices, :] = right
        # points_2D[:, cam, right_indices, :] = left

        cropzone = h5py.File(box_path, 'r')['/cropzone'][:-1]
        points_3D_all_multiviews, reprojection_errors_multiviews = tr.triangulate_points_all_possible_views(points_2D, cropzone)
        points_3D_all, reprojection_errors, triangulation_errors = tr.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)
        # points_3D_all_rays_midpoint, reprojection_errors_rays_midpoint, triangulation_errors_rays_midpoint = tr.rays_midpoint_traingulation(points_2D, cropzone)
        mean_reprojection_error = np.mean(reprojection_errors, axis=(0, 1))
        print(mean_reprojection_error)
        mean_3D_error = np.mean(triangulation_errors, axis=(0, 1))
        reprojection_errors_per_pair = np.reshape(reprojection_errors, [-1, 6])
        pass
