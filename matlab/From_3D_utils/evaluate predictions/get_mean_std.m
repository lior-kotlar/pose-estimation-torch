function mean_std = get_mean_std(points_3D)
    [~, dist_variance, ~]= get_wing_distance_variance(points_3D); 
    mean_std = mean(sqrt(dist_variance)); 