function [predictions, box] = fix_wings_3d(predictions, easyWandData, cropzone, box, flip_box)
%[cam1, cam2, cam3]
which_to_flip = [[0,0,0];[0,0,1];[0,1,0];[0,1,1];[1,0,0];[1,0,1];[1,1,0];[1,1,1]];
num_of_options = size(which_to_flip, 1);
scores = zeros(num_of_options, 1);
test_preds = predictions(:, :,:,:);
num_frames = size(test_preds, 1);
num_joints = size(test_preds, 3);
cam_inds=1:size(predictions,2);
left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
for option=1:size(which_to_flip, 1)
    test_preds_i = test_preds;
    cams_to_flip = which_to_flip(option, :);
    % flip right and left indexes in relevant cameras
    for cam=1:3
        if cams_to_flip(cam) == 1
            left_wings_preds = squeeze(test_preds(:, cam + 1, left_inds, :));
            right_wings_preds = squeeze(test_preds(:, cam + 1, right_inds, :));
            test_preds_i(:, cam + 1, right_inds, :) = left_wings_preds;
            test_preds_i(:, cam + 1, left_inds, :) = right_wings_preds;
        end
    end
    % get 6 3d pts for every joint
    [~, ~ ,test_3d_pts] = get_3d_pts_rays_intersects(test_preds_i, easyWandData, cropzone, cam_inds);
    % compute box volume 
    total_boxes_volume = 0;
    for frame=1:num_frames
        for pnt=1:num_joints
            joint_pts = squeeze(test_3d_pts(pnt, frame,: ,:)); 
            cloud = pointCloud(joint_pts);
            x_limits = cloud.XLimits;
            y_limits = cloud.YLimits;
            z_limits = cloud.ZLimits;
            x_size = abs(x_limits(1) - x_limits(2)); 
            y_size = abs(y_limits(1) - y_limits(2));
            z_size = abs(z_limits(1) - z_limits(2));
            box_volume = x_size*y_size*z_size;
            total_boxes_volume = total_boxes_volume + box_volume;
        end
    end
    avarage_box_volume = total_boxes_volume/(num_joints*num_frames);
    scores(option) = avarage_box_volume;
end
[M,I] = min(scores,[],'all');
winning_option = which_to_flip(I, :);
aranged_predictions = predictions;
for cam=1:3
    if winning_option(cam) == 1
        % switch left and right prediction indexes
        left_wings_preds = squeeze(predictions(:, cam + 1, left_inds, :));
        right_wings_preds = squeeze(predictions(:, cam + 1, right_inds, :));
        aranged_predictions(:, cam + 1, right_inds, :) = left_wings_preds;
        aranged_predictions(:, cam + 1, left_inds, :) = right_wings_preds;
    end
end
predictions = aranged_predictions;
% deal with wings masks
% display_predictions_2D(box,predictions, 0);
if flip_box
    if num_joints > 2
        frame = 1;
        for cam=1:4
            right_wing_preds = squeeze(predictions(frame, cam, right_inds, :));
            count = 0;
            right_wing_mask = squeeze(box(:, :, 3, cam, frame));
            for joint=1:size(right_wing_preds,1)
                count = count + right_wing_mask(right_wing_preds(joint, 2), right_wing_preds(joint, 1)); 
            end
        %     figure; imshow(squeeze(box(:, :, :, cam, frame)));
        %     hold on 
        %     x = right_wing_preds(:,1);
        %     y = right_wing_preds(:,2);
        %     scatter(x, y, 44, 'LineWidth',3);
            if count < 3
                % switch first and second masks
                first_masks = box(:, :, 2, cam, :);
                second_masks = box(:, :, 3, cam, :);
                box(:, :, 2, cam, :) = second_masks;
                box(:, :, 3, cam, :) = first_masks;
            end
        end
    else
        num_test_frames=20;
        bad_count=0;
        for cam=1:4
            for frame=1:num_test_frames
                left_wing_mask = squeeze(box(:, :, 2, cam, frame));
                right_wing_mask = squeeze(box(:, :, 3, cam, frame));
                left_pt = squeeze(predictions(frame, cam, 1, :));
                right_pt = squeeze(predictions(frame, cam, 2, :));
                
                dist_r2r = distance_from_mask_to_point(right_wing_mask, right_pt);
                dist_r2l = distance_from_mask_to_point(right_wing_mask, left_pt);
                if dist_r2l < dist_r2r
                    bad_count = bad_count + 1;
                end
            end
            if bad_count > 15
                % switch wings
                first_masks = box(:, :, 2, cam, :);
                second_masks = box(:, :, 3, cam, :);
                box(:, :, 2, cam, :) = second_masks;
                box(:, :, 3, cam, :) = first_masks;
            end
        end
      
    end
end
end

function dist = distance_from_mask_to_point(mask, point)
    D = bwdist(mask);
    dist = D(point(1), point(2));
end
