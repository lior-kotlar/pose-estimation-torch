function [GT_2D, GT_3D] = get_ground_truth_labels(cropzone, easyWandData, box)
    %% load ground truth
    num_labeled = 70;
    labeled_indx = 1:num_labeled;
    GT_path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\movie\movie_1_1701_2200_500_frames_3tc_7tj.labels.mat";
    a = load(GT_path);
    GT_2D = a.positions(:,:,:,labeled_indx);
    GT_2D = permute(GT_2D, [4,3,1,2]);
    %% get 3D of labeled
    [all_errors_70, ~, all_pts3d_70] = get_3d_pts_rays_intersects(GT_2D, easyWandData, cropzone(:,:,labeled_indx), 1:4);
    [GT_3D, ~] = get_3D_pts_2_cameras(all_pts3d_70, all_errors_70, GT_2D, box);
    GT_3D = squeeze(GT_3D);
end