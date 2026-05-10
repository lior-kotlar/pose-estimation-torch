function [all_pts_3d, all_errors] = triangulate_points(points_2d, easyWandData, cropzone)
    allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
    rotation_mat = allCams.Rotation_Matrix';
    num_joints = size(points_2d, 3);
    num_cams = size(points_2d, 2);
    points_2D = permute(points_2d, [3, 2, 1, 4]);
    % trasnfer to original image coordinates
    for joint=1:num_joints
        for cam=1:num_cams
            xs = double(squeeze(cropzone(2,cam,:))) + squeeze(points_2D(joint,cam, :, 1));
            ys = double(squeeze(cropzone(1,cam,:))) + squeeze(points_2D(joint,cam, :, 2));
            ys = 801 - ys;
            points = [xs, ys];
            points_2D(joint, cam, :, :) = points;
        end
    end
    projection_matrices = zeros(num_cams, 3, 4);
    for cam=1:num_cams
        projection_matrices(cam, :, :) = allCams.cams_array(cam).reshaped_dlt;
    end

    addpath("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation");
    
    % save the arguments
    save('C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\variables.mat', ...
    'points_2d', 'projection_matrices', 'rotation_mat');
    
    % run python file to traingulate in 
    system("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\venv\Scripts\python.exe " + ...
    "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\triangulation_2D_3D.py")

    data_1 = load("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\all_pts_3d.mat");
    data_2 = load("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\all_errors.mat");
    all_pts_3d = data_1.all_pts_3d;
    all_pts_3d = permute(all_pts_3d, [3, 1, 2, 4]);
    all_errors = data_2.all_errors;
    all_errors = permute(all_errors, [3, 1, 2]);
end