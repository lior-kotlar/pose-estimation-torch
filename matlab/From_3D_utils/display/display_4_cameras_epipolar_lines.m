function display_4_cameras_epipolar_lines(points_3D, points_2D, chosen_pt_3D, cropzone, easyWandData)
% UNTITLED12 Summary of this function goes here
% points_2D : array of size (4, 2)
% points_3D : array of size (11, 3) all candidates for point
% cropzone  : array of size (2, 4)
% chosen_pt_3D: array of size (1, 3)
allCams = HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
num_cams = length(allCams.cams_array);
centers=allCams.all_centers_cam';
cam_inds=1:num_cams;
couples=nchoosek(1:num_cams,2);
num_couples=size(couples,1);
figure;
for cam=1:num_cams
    x = double(cropzone(2, cam))+squeeze(points_2D(cam,1));
    y = double(cropzone(1, cam))+squeeze(points_2D(cam,2));
    PB_cam = allCams.cams_array(cam_inds(cam)).invDLT * [x; (801-y); 1];
    normalized_PB = PB_cam(1:3,:)./PB_cam(4,:);
    PB(cam,:) = normalized_PB;
    center = centers(cam, :);
    Ps = allCams.Rotation_Matrix * normalized_PB;
    allPts=[allCams.Rotation_Matrix*allCams.cams_array(cam).camera_cnt,Ps];
    x_3d = allPts(1, :);
    y_3d = allPts(2, :);
    z_3d = allPts(3, :);
    plot3(x_3d,y_3d,z_3d, 'Color', circshift([1,0,0],cam), 'LineWidth',1);
    hold on 
end
% scatter all pts
scatter3(points_3D(:, 1),points_3D(:, 2),points_3D(:, 3));
% scatter chosen pt
scatter3(chosen_pt_3D(1),chosen_pt_3D(2),chosen_pt_3D(3))
max_lims = mean(points_3D) + 0.0005; 
min_lims = mean(points_3D) - 0.0005;
xlim([min_lims(1), max_lims(1)]);
ylim([min_lims(2), max_lims(2)]);
zlim([min_lims(3), max_lims(3)]);
grid on
a=0;
end