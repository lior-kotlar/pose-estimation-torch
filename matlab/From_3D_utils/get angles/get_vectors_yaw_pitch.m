function [yaw_angle, pitch_angle] = get_vectors_yaw_pitch(vectors)
    % vector of (2, N, 3) 
    x=1;y=2;z=3;
    norm_vec = get_head_tail_vec_all(vectors);
    dx = norm_vec(:, x);
    dy = norm_vec(:, y);
    dz = norm_vec(:, z);
    yaw_angle = rad2deg(atan2(dy, dx));
    pitch_angle = rad2deg(atan2(dz, sqrt(power(dx, 2) + power(dy, 2))));

end