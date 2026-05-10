function [mean_std, dist_GT_3D, dist_GT_2D] = get_preds_evaluation(GT_2D, GT_3D, preds_2D, preds_3D)
    mean_std = get_mean_std(preds_3D(1:14, :, :));
    GT_2D = squeeze(GT_2D); GT_3D = squeeze(GT_3D);
    preds_2D = squeeze(preds_2D); preds_3D = squeeze(preds_3D);
    num_joints = min(size(preds_3D,1) ,size(GT_3D,1));
    num_frames = min(size(preds_3D,2) ,size(GT_3D,2));
    dist_GT_2D = get_all_points_distances(preds_2D(1:num_frames, :, 1:num_joints, :), GT_2D(1:num_frames, :, 1:num_joints, :));
    dist_GT_3D = get_all_points_distances(preds_3D(1:num_joints, 1:num_frames, :), GT_3D(1:num_joints, 1:num_frames, :));
    mead_dist_2D = mean(dist_GT_2D);
    mean_dist_3D = mean(dist_GT_3D);
%     histogram(dist_GT_2D)
%     histogram(dist_GT_3D)
    a=0;
end