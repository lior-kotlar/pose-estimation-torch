function wing_predictions_fixed = fix_predictions_wings(wing_predictions, easyWandData, cropzone)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    which_to_flip = [[0,0,0];[0,0,1];[0,1,0];[0,1,1];[1,0,0];[1,0,1];[1,1,0];[1,1,1]];
    num_of_options = size(which_to_flip, 1);
    num_frames = size(wing_predictions, 1);
    num_joints = size(wing_predictions, 3);
    cam_inds = 1:size(wing_predictions,2);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    wing_predictions_fixed = nan(size(wing_predictions));
    for frame=1:num_frames
        pts_2D_i = wing_predictions(frame, :,:,:);
        crop_i = cropzone(:,:,frame);
        scores = zeros(num_of_options, 1);
        for option=1:num_of_options
            cams_to_flip = which_to_flip(option, :);
            test_preds_i = pts_2D_i;
            % flip cameras
            for cam=1:3
                if cams_to_flip(cam) == 1
                    left_wings_preds = squeeze(pts_2D_i(:, cam + 1, left_inds, :));
                    right_wings_preds = squeeze(pts_2D_i(:, cam + 1, right_inds, :));
                    test_preds_i(:, cam + 1, right_inds, :) = left_wings_preds;
                    test_preds_i(:, cam + 1, left_inds, :) = right_wings_preds;
                end
            end
            % get 6 3d pts for every joint
            [~, ~ ,all_pts_3d_test] = get_3d_pts_rays_intersects(test_preds_i, easyWandData, cropzone, cam_inds);
            total_boxes_volume = 0;
            for pnt=1:num_joints
                joint_pts = squeeze(all_pts_3d_test(pnt, :,: ,:)); 
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
            avarage_box_volume = total_boxes_volume/(num_joints);
            scores(option) = avarage_box_volume;
        end
        [M,I] = min(scores,[],'all');
        winning_option = which_to_flip(I, :);
        new_predictions_2D = wing_predictions(frame, :,:,:);
        for cam=1:3
            if winning_option(cam) == 1
                % switch left and right indexes
                left_wings_preds = squeeze(new_predictions_2D(:, cam + 1, left_inds, :));
                right_wings_preds = squeeze(new_predictions_2D(:, cam + 1, right_inds, :));
                new_predictions_2D(:, cam + 1, right_inds, :) = left_wings_preds;
                new_predictions_2D(:, cam + 1, left_inds, :) = right_wings_preds;
            end
        end
        wing_predictions_fixed(frame,:,:,:) = new_predictions_2D;
    end
end




