function body_roll = get_body_roll(wings_joints_pts) 
    [~, body_roll] = get_vectors_yaw_pitch(wings_joints_pts);
end