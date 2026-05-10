function [preds_2D, box] = fix_wings_3d_body_parts(wings_joints_2D, easyWandData, cropzone, box)
    which_to_flip = [[0,0,0];[0,0,1];[0,1,0];[0,1,1];[1,0,0];[1,0,1];[1,1,0];[1,1,1]];
    num_of_options = size(which_to_flip, 1);
    scores = zeros(num_of_options, 1);
    test_preds = wings_joints_2D(:, :,:,:);
    num_frames = size(test_preds, 1);
    num_joints = size(test_preds, 3);
    cam_inds=1:size(wings_joints_2D,2);
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
    aranged_predictions = wings_joints_2D;
    for cam=1:3
        if winning_option(cam) == 1
            % switch left and right prediction indexes
            left_wings_preds = squeeze(wings_joints_2D(:, cam + 1, left_inds, :));
            right_wings_preds = squeeze(wings_joints_2D(:, cam + 1, right_inds, :));
            aranged_predictions(:, cam + 1, right_inds, :) = left_wings_preds;
            aranged_predictions(:, cam + 1, left_inds, :) = right_wings_preds;
        end
    end
    preds_2D = aranged_predictions;
end