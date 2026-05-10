function theta = get_theta(strok_plane_normal, wing_vec)
    % returns theta (the z axis angle) between waving_axis, wing_vec 
    % assumes both vectors are normalized
    dp = dot (strok_plane_normal, wing_vec);
    theta = acos(dp);
    theta = 90 - rad2deg(theta);  % could be also just the cos-1 
end