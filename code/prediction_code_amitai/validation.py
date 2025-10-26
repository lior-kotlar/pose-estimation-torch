import numpy as np
from scipy.spatial import distance_matrix

PAIRS_OF_NEIGHBORS = [(0,1), (1,2), (2,3), (3,4),   (4,5),   (5,6),  (0,6),
                      (7,8), (8,9), (9,10),(10,11), (11,12), (12,13),(7,13),
                      (14,15)]

POINTS_DICT = {0: [0, 6], 1: [0, 1], 2: [1, 2], 3: [2, 3], 4: [3, 4], 5: [4, 5], 6: [5, 6],
               7: [7, 13], 8: [7, 8], 9: [8, 9], 10: [9, 10], 11: [10, 11], 12: [11, 12], 13: [12, 13]}

# for i, pair in enumerate(PAIRS_OF_NEIGHBORS[:-1]):
#     pairs_inds = points_dict[i]
#     print(f"{i}, {PAIRS_OF_NEIGHBORS[pairs_inds[0]]}, {PAIRS_OF_NEIGHBORS[pairs_inds[1]]}")
#


class Validation:
    """
    A class of validation methods
    """
    @staticmethod
    def get_wings_distances_variance(points_3d):
        """

        Args:
            points_3d: (num_frames, num_joints, 3) array

        Returns:
            mean_distances: mean_distances[i] mean distance from joint i to joint i+1
            distances_variance: distances_variance[i] variance of distance from point i to point i+1
            points_distances: the distance from point i to point i+1 in 'frame'
                distances: [0] 0 -> 1, [1] 1 -> 2, [2] 2 -> 3, [3] 3 -> 4, [4] 4 -> 5, [5] 5 -> 6, [6] 6 -> 0
                           [7] 7 -> 8, [8] 8 -> 9, [9] 9 -> 10, [10] 10 -> 11, [11] 11 -> 12, [12] 12 -> 13, [13] 13->0
                           [14] 14 -> 15
            average_std: the average std of the points' distances
        """
        if points_3d.shape[1] == 18:
            points_3d = points_3d[:, [x for x in range(18) if x not in [7, 15]], :]
        num_frames = points_3d.shape[0]
        num_wings_pnts = points_3d.shape[1] - 2
        points_per_wing = num_wings_pnts // 2
        points_distances = np.zeros((num_frames, 2 * points_per_wing + 1))
        for frame in range(num_frames):
            points = points_3d[frame, ...]
            distances, _ = Validation.get_points_distances(points)
            points_distances[frame, :] = distances
        mean_distance = np.mean(points_distances, axis=0)
        median_distance = np.median(points_distances, axis=0)

        distances_std = points_distances.std(axis=0)

        # for each point get the median distance from it to it's 2 neighbors
        neighbors_dist = np.zeros((num_wings_pnts, 2, 2))
        for pnt in range(num_wings_pnts):
            for neighbor in range(2):
                ind = POINTS_DICT[pnt][neighbor]
                mean_dist = mean_distance[ind]
                std = distances_std[ind]
                neighbors_dist[pnt, neighbor, 0] = mean_dist
                neighbors_dist[pnt, neighbor, 1] = std

        average_std = distances_std.mean()
        return average_std, neighbors_dist, distances_std, mean_distance, points_distances

    @staticmethod
    def get_points_distances(points):
        distances = np.zeros(len(PAIRS_OF_NEIGHBORS), )
        for i, pair in enumerate(PAIRS_OF_NEIGHBORS):
            pnt_a, pnt_b = points[pair[0], :], points[pair[1], :]
            dist = np.linalg.norm(pnt_a - pnt_b)
            distances[i] = dist
        num_wings_pnts = (len(points) - 2)
        neighbors_dist = np.zeros((len(points) - 2, 2))
        for pnt in range(num_wings_pnts):
            ind_a, ind_b = POINTS_DICT[pnt]
            med_dist_a = distances[ind_a]
            med_dist_b = distances[ind_b]
            neighbors_dist[pnt, 0] = med_dist_a
            neighbors_dist[pnt, 1] = med_dist_b
        return distances, neighbors_dist
