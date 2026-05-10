function points_3D = get_best_2D_loss_pts_3D(all_pts_3D, predictions_2D, easyWandData, cropzone)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
n_frames = size(predictions_2D, 1);
num_joints = size(predictions_2D, 3);
num_cams = length(allCams.cams_array);
num_candidates = size(all_pts_3D, 3);
points_3D = nan(num_joints, n_frames, 3);
for frame=1:n_frames
    for joint=1:num_joints
        candidates = squeeze(all_pts_3D(joint,frame,:,:)); 
        scores = nan(num_candidates, num_cams);
        for cand=1:num_candidates
            pt_3D = squeeze(candidates(cand, :));
            for cam=1:num_cams
                joint_pt = allCams.Rotation_Matrix' * pt_3D';
                xy_per_cam_per_joint = dlt_inverse(allCams.cams_array(cam).dlt, joint_pt');
                % flip y
                xy_per_cam_per_joint(2) = 801 - xy_per_cam_per_joint(2);
                x_p = xy_per_cam_per_joint(1); y_p = xy_per_cam_per_joint(2);
                % crop
                x_crop = cropzone(2, cam, frame);
                y_crop = cropzone(1, cam, frame);
                x_cand = double(x_p - x_crop); 
                y_cand = double(y_p - y_crop);
                x_pred = squeeze(predictions_2D(frame, cam, joint, 1));
                y_pred = squeeze(predictions_2D(frame, cam, joint, 2));
                pt_error = abs(x_cand - x_pred) + abs(y_cand - y_pred); 
                scores(cand, cam) = pt_error;
            end
        end
        % get rid of worse camera
        scores = sort(scores, 2);
        scores = scores(:, 1:(num_cams - 2));
        mean_scores = mean(scores, 2);
        [M,I] = min(mean_scores, [], 'all');
        best_pt = candidates(I, :);
        points_3D(joint, frame, :) = best_pt;
    end
end
end