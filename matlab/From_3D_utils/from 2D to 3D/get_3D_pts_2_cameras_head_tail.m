function  [head_tail_pts, head_tail_errors] = get_3D_pts_2_cameras_head_tail(all_pts3d, all_errors, predictions, box, body_masks)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    num_cams_to_use=2;
    num_joints=size(predictions,3);
    head = 1; tail = 2;
    n_frames = size(predictions,1);
    num_cams = size(predictions,2);
    all_cams = 1:num_cams;
    num_couples = nchoosek(num_cams_to_use,2);
    all_couples = nchoosek(1:num_cams, 2);
    head_tail_pts = nan(num_joints,n_frames,3);
    head_tail_errors = nan(num_joints,n_frames);
    [preds_2D_derivative, envelope_2D] = get_2D_derivative(predictions);
    for frame_ind=1:n_frames
        if frame_ind == 6
            a=0;
        end
        for node_ind=1:num_joints
            % find best cameras
            % pixels with body and other wing
            masks = squeeze(body_masks(:, :, :, frame_ind));
            masks_sizes = zeros(num_cams, 1);
            for mask_num=1:num_cams
                mask = squeeze(masks(:,:,mask_num));
                mask_size = nnz(mask);
                masks_sizes(mask_num) = mask_size;
            end
            %% get a score by a*big_masks (a-1)*not_noisy 
            pnt_noise_val = squeeze(envelope_2D(frame_ind, :, node_ind));
            pnt_noise_val = 1 - pnt_noise_val/max(pnt_noise_val);
            pnt_noise_val(isnan(pnt_noise_val)) = 0;
            masks_sizes_norm = masks_sizes/max(masks_sizes);

            alpha = 0.5;
            score_vals = alpha*masks_sizes_norm' + (1-alpha)*pnt_noise_val;

            [sorted_array, sorted_indexes] = sort(score_vals, 'descend');
            cam_inds = sorted_indexes(1:num_cams_to_use);
            
            pt_ind = all(ismember(all_couples, cam_inds), 2);
            pnt = squeeze(all_pts3d(node_ind, frame_ind, pt_ind, :));
            error = squeeze(all_errors(node_ind, frame_ind, pt_ind));
            head_tail_pts(node_ind, frame_ind, :) = pnt;
            head_tail_errors(node_ind, frame_ind) = error;
        end
    end

end