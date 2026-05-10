function stroke_planes = get_stroke_planes(points_3D)
% points_3D get 18 points
num_frames = size(points_3D, 2);
head_tail_pts = points_3D([17,18], :, :);
head_tail_vec = get_head_tail_vec_all(head_tail_pts);
wings_joints = points_3D([15,16], :, :);
ybody = get_y_body(wings_joints);
% find the bormal to the stroke plane
theta = pi/4; 
stroke_normal = rodrigues_rot(head_tail_vec, ybody, theta);
point_on_plane = squeeze(mean(wings_joints, 1));
d_s = -dot(ybody, point_on_plane, 2);  % ax + by + cz + d
stroke_planes = cat(2, stroke_normal, d_s);
end
