function wings_roll = get_wings_roll(points_3D)
    x=1;y=2;z=3;
    num_wing_pts = 14;
    pnts_per_wing = num_wing_pts/2;
    points_3D = squeeze(points_3D);
    num_frames = size(points_3D, 2);
    num_joints = size(points_3D,1);
    left_inds = 1:(num_wing_pts/2); 
    right_inds = (num_wing_pts/2 + 1:num_wing_pts);
    num_wings = 2;
    %% get the strok planes
    stroke_planes = get_stroke_planes(points_3D);
    %% get all other wings planes 
    [all_planes_dists, all_4_planes, all_2_planes] = get_all_planes(points_3D);
    
    %% get roll angles
    num_of_planes_per_wing = 3;
    wings_roll = zeros(num_wings, num_frames, num_of_planes_per_wing);
    % the angle between the strok plane normal and each of this normals
    % all wing plane
    left_wing_planes = squeeze(all_2_planes(1, :, :));
    right_wing_planes = squeeze(all_2_planes(2, :, :));
    left_wing_roll = get_angle_between_planes(left_wing_planes, stroke_planes);
    right_wing_roll = get_angle_between_planes(right_wing_planes, stroke_planes);
    
    % upper and lower wing's planes
    upper_left_wing_planes = squeeze(all_4_planes(1, :, :));
    lower_left_wing_planes = squeeze(all_4_planes(2, :, :));
    upper_right_wing_planes = squeeze(all_4_planes(3, :, :));
    lower_right_wing_planes = squeeze(all_4_planes(4, :, :));

    upper_left_wing_roll = get_angle_between_planes(upper_left_wing_planes, stroke_planes);
    lower_left_wing_roll = get_angle_between_planes(lower_left_wing_planes, stroke_planes);
    upper_right_wing_roll = get_angle_between_planes(upper_right_wing_planes, stroke_planes);
    lower_right_wing_roll = get_angle_between_planes(lower_right_wing_planes, stroke_planes);

    wings_roll(1, :, 1) = left_wing_roll;
    wings_roll(1, :, 2) = upper_left_wing_roll;
    wings_roll(1, :, 3) = lower_left_wing_roll;
    wings_roll(2, :, 1) = right_wing_roll;
    wings_roll(2, :, 2) = upper_right_wing_roll;
    wings_roll(2, :, 3) = lower_right_wing_roll;
end

function angle = get_angle_between_planes(plane1, plane2)
    P1_normal = plane1(:, [1,2,3]);
    P2_normal = plane2(:, [1,2,3]);
    rad = acos(dot(P1_normal, P2_normal, 2));
    angle = rad2deg(rad);
end