function [errors_3D, preds_3D, preds_2D, box] = get_predictions_2D_3D(body_parts_path ,wings_preds_path, easy_wand_path)
    % get the 2D and 3D points of predictions 
    box = h5read(wings_preds_path,'/box');
    box = reshape_box(box, 1);
    cropzone = h5read(wings_preds_path,'/cropzone');
    num_wings_pts = 14;
    
    % seg_scores = permute(h5read(wings_preds_path,'/scores'), [3,2,1]);
    wings_preds = h5read(wings_preds_path,'/positions_pred');
    wings_preds = single(wings_preds) + 1;
    
    body_preds = h5read(body_parts_path,'/positions_pred');
    body_preds = single(body_preds) + 1;
    num_body_pts = size(body_preds, 1);

    num_wings_pts = size(wings_preds,1);
    pnt_per_wing = num_wings_pts/2;
    left_inds = 1:pnt_per_wing; right_inds = (pnt_per_wing+1:num_wings_pts); 
    wings_joints_inds = (num_wings_pts + num_body_pts - 3):(num_wings_pts + num_body_pts - 2);
    head_tail_inds = (num_wings_pts + num_body_pts - 1):(num_wings_pts + num_body_pts);
    body_parts_inds = (num_wings_pts + 1):(num_wings_pts + num_body_pts);
    easyWandData=load(easy_wand_path);
    allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
    num_cams=length(allCams.cams_array);
    cam_inds=1:num_cams;
    num_frames=size(box, 5);
    wings_inds = [1:num_wings_pts/2, wings_joints_inds(1),(num_wings_pts/2 + 1):num_wings_pts, wings_joints_inds(2)];
    x=1; y=2; z=3;

    %% rearange predictions wings
    if ndims(wings_preds) == 4
        wings_preds_2D = permute(wings_preds, [4, 3, 1, 2]);
        % wings_preds_2D_x = wings_preds_2D(:, :, :, 1);
        % wings_preds_2D_y = wings_preds_2D(:, :, :, 2);
        % wings_preds_2D(:, :, :, 1) = wings_preds_2D_y;
        % wings_preds_2D(:, :, :, 2) = wings_preds_2D_x;
    else
        wings_preds_2D = rearange_predictions(wings_preds, num_cams);
    end
        
    %% rearange predictions body
    if ndims(body_preds) == 4
        body_parts_2D = permute(body_preds, [4,3,1,2]);
    else
        body_parts_2D = rearange_predictions(body_preds, num_cams);
    end
    head_tail_preds_2D = body_parts_2D(:,:,3:4, :);
    
    %% fix predictions per camera 
    [wings_preds_2D, box] = fix_wings_per_camera(wings_preds_2D, box);
    
    %% fix wing 1 and wing 2 
    [wings_preds_2D, box] = fix_wings_3d(wings_preds_2D, easyWandData, cropzone, box, true);

    %% fix wings in one function
    % [wings_preds_2D, box, body_parts_2D, seg_scores] = fix_wings_3d_per_frame(wings_preds_2D, body_parts_2D, easyWandData, cropzone, box, seg_scores);
    
    %% find wing joints sides
    for frame=1:num_frames
        for cam=1:num_cams
             left_wj = squeeze(body_parts_2D(frame, cam, 1, :)); 
             right_wj = squeeze(body_parts_2D(frame, cam, 2, :)); 

             left_mask = squeeze(box(:,:,2,cam, frame));
             right_mask = squeeze(box(:,:,3,cam, frame));
             dist_r2r  = get_distance_from_mask_to_point(right_mask, right_wj);
             dist_r2l = get_distance_from_mask_to_point(right_mask, left_wj);
             dist_l2l  = get_distance_from_mask_to_point(left_mask, left_wj);
             dist_l2r = get_distance_from_mask_to_point(left_mask, right_wj);
             if dist_r2l < dist_r2r || dist_l2r < dist_l2l
                % switch body points location 
                body_parts_2D(frame, cam, 1, :) = right_wj;
                body_parts_2D(frame, cam, 2, :) = left_wj;
             end
        end
    end


    %% join all points 
    preds_2D(:,:,1:num_wings_pts,:) = wings_preds_2D;
    preds_2D(:,:,body_parts_inds,:) = body_parts_2D;
    %% display 
    % display_box = view_masks_perimeter(box);
    % display_predictions_2D_tight(display_box, preds_2D, 0) 

    %% get 3d pts from 4 2d cameras 
    [all_errors, ~, all_pts3d] = get_3d_pts_rays_intersects(preds_2D, easyWandData, cropzone, cam_inds);
     %%
                                                    % untill here all is well
     %%
    %% get avarage consecpoints 
%     rem_outliers = 1;
%     ThresholdFactor = 0.2;
%     avg_consecutive_pts = get_avarage_consecutive_pts(all_pts3d, rem_outliers, ThresholdFactor);

    %% get head tail pts 3D
    all_pts3d_head_tail = all_pts3d(head_tail_inds, :, :, :);
    all_errors_head_tail = all_errors(head_tail_inds, :, :);
    body_masks = get_body_masks(wings_preds_path, 0);

    head_tail_all_ptse3d = all_pts3d(head_tail_inds,:,:,:);
    head_tail_errors = all_errors(head_tail_inds, :,:,:);
%     head_tail_pts = get_avarage_points_3d(all_pts3d_head_tail, all_errors_head_tail, 1, 1.3);
    head_tail_pts = get_3D_pts_2_cameras_head_tail(head_tail_all_ptse3d, head_tail_errors, head_tail_preds_2D, box, body_masks);
        
    %% get wing joing points 3D
    all_wings_joints_pts = all_pts3d(wings_joints_inds, :, :, :);
    all_errors_wings_joints = all_errors(wings_joints_inds, :, :);
    wings_joints_pts = get_avarage_points_3d(all_wings_joints_pts, 1, 1.3);
    
    %% get 3D
    erosion_rad = 0;
    body_masks = get_body_masks(wings_preds_path, erosion_rad);
    only_wings_predictions = preds_2D(:,:,wings_inds,:);
    only_wings_all_pts3d = all_pts3d(wings_inds,:,:,:);
    [preds_3D, errors_3D] = get_3D_pts_2_cameras(only_wings_all_pts3d, all_errors, only_wings_predictions, box, body_masks);
%     [preds_3D, errors_3D]  = get_3D_pts_2_cameras_smart(only_wings_all_pts3d, all_errors, only_wings_predictions, box, body_masks);
    
    % rearrange preds_3D 
    wings_joints_pts = preds_3D([8, 16], :,: );
    left_wing_pnts = preds_3D((1:7), :,: );
    right_wing_pnts = preds_3D((9:15), :,: );

    preds_3D(1:7, :, :) = left_wing_pnts;
    preds_3D(8:14, :, :) = right_wing_pnts;
    preds_3D([15, 16], :, :) = wings_joints_pts;

    preds_3D(head_tail_inds, :,:) = head_tail_pts;
    preds_3D = squeeze(preds_3D);
%     display_predictions_pts_3D(preds_3D, 0.1);

    %% do postproccesing 3D points for correction
%     preds_3D = post_correct(preds_3D, all_pts3d);
end