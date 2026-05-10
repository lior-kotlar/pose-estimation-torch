function cameras_used = get_2_cameras_per_frame_per_wing(box, body_masks)
    num_frames = size(box, 5);
    num_cams = size(box, 4);
    num_cams_to_use=2;
    for frame=1:num_frames
        for wing = 1:2
            if wing == 2 othe_wing = 1; elseif wing == 1 other_wing = 2; end
            masks = squeeze(box(:,:,1 + wing,:,frame));
            other_masks = squeeze(box(:,:,1 + other_wing,:,frame));
            masks_sizes = zeros(num_cams, 1);
            for mask_num=1:num_cams
                mask = squeeze(masks(:,:,mask_num));
                body = squeeze(body_masks(:, :, mask_num, frame));
                other_mask = squeeze(other_masks(:,:,mask_num));
                body_and_sec_wing = other_mask | body;
                intersection = mask & body_and_sec_wing;
                mask_min_body = mask - intersection;
                mask_size = nnz(mask_min_body);
                masks_sizes(mask_num) = mask_size;
            end
            [sorted_array, sorted_indexes] = sort(masks_sizes, 'descend');
            cam_inds = sorted_indexes(1:num_cams_to_use);
            cameras_used(frame, wing, :) = cam_inds;
        end
    end
end