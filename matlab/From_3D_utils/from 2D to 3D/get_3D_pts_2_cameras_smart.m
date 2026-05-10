function [preds_3D, errors_3D] = get_3D_pts_2_cameras_smart(all_pts3d ,all_errors, predictions, box, body_masks)
    % returns a 3D point for each node in a frame, based on choosig the
    % best 2 cameras
    % cameras are chosen by size of masks, as well as if the point is
    % hidden by flie's body
%     body_masks = get_body_masks(wings_preds_path);
%     box = reshape_box(h5read(wings_preds_path,'/box'), 1);
    num_cams_to_use=2;
    num_joints=size(predictions,3);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    n_frames = size(predictions,1);
    num_cams = size(predictions,2);
    preds_3D = nan(num_joints,n_frames,3);
    errors_3D = nan(num_joints,n_frames);
    for frame_ind=1:n_frames
        for node_ind=1:num_joints
            % find best cameras
            if ismember(node_ind, left_inds)
                masks = squeeze(box(:,:,2,:,frame_ind));
            elseif ismember(node_ind, right_inds)
                masks = squeeze(box(:,:,3,:,frame_ind));
            end
            % find the largest num_cams cameras can be done before
            masks_sizes = zeros(num_cams, 1);
            for mask_num=1:num_cams
                mask = squeeze(masks(:,:,mask_num));
                mask_size = nnz(mask);
%                 mask_intersec_with_body = mask & squeeze(body_masks(:,:,mask_num, frame_ind));
                masks_sizes(mask_num) = mask_size;
            end
            
            %% remove camera in which body is in the way
            bad_cams = zeros(num_cams, 1);
            for cam=1:num_cams
                pnt = squeeze(predictions(frame_ind, cam, node_ind, :));
                body_mask = squeeze(body_masks(:,:,cam, frame_ind));
                p_x = pnt(1);
                p_y = pnt(2);
%                 figure; imshow(body_mask); hold on; scatter(p_x, p_y, 'red');
                if body_mask(p_y, p_x) == 1
                    bad_cams(cam) = 1;
                end
            end
            masks_sizes(bad_cams == 1) = 0;
            
            %% take 3D points
            [sorted_array, sorted_indexes] = sort(masks_sizes, 'descend');
            all_masks_sizes(frame_ind,:,:) = sorted_array;
            cam_inds = sorted_indexes(1:num_cams_to_use);
            all_couples = nchoosek(1:num_cams, 2);
            pt_ind = all(ismember(all_couples, cam_inds), 2);
            pnt = squeeze(all_pts3d(node_ind, frame_ind, pt_ind, :));
            error = squeeze(all_errors(node_ind, frame_ind, pt_ind));
            preds_3D(node_ind, frame_ind, :) = pnt;
            errors_3D(node_ind, frame_ind) = error;
        end
    end
end