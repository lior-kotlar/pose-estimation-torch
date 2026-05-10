function wing_template = create_wing_template(wing_pts)
% wing_pts : (pts_per_wing, num_frames, 3) array of points
% wing_template : (pts_per_wing, 3) avarage wing template
reference_wing = squeeze(wing_pts(:, 35, :));
num_frames = size(wing_pts, 2);
new_points = zeros(size(wing_pts));
for frame=1:500
    fixed = pointCloud(reference_wing);
    moving = pointCloud(squeeze(wing_pts(:, frame, :)));
    [T_matrix, registered_wing] = pcregrigid(moving, fixed);
    new_points(:, frame, :) = registered_wing.Location;
end
% plot3(squeeze(new_points(:, :, 1)), squeeze(new_points(:, :, 2)),squeeze(new_points(:, :, 3)), 'o')
wing_template = squeeze(mean(new_points, 2)); 
plot3(squeeze(wing_template(:, 1)), squeeze(wing_template(:, 2)),squeeze(wing_template(:, 3)), 'o-r')
grid on
end