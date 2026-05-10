function ybody = get_y_body(wings_joints_3D)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
left_pnt = squeeze(wings_joints_3D(1, : ,:));
right_pnt = squeeze(wings_joints_3D(2, : ,:));
ybody = normr(left_pnt - right_pnt);
end

