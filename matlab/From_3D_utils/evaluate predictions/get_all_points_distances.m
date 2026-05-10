function distance_vector = get_all_points_distances(points_1, points_2)
    dim = size(points_1, ndims(points_1));
    flattend_points_1 = double(reshape(points_1 ,[], dim));
    flattend_points_2 = double(reshape(points_2, [], dim));
    distance_vector = vecnorm(flattend_points_1 - flattend_points_2, 2, 2);
