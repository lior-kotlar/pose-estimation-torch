function [smoothed_pts] = smooth_by_avaraging(points_to_smoooth,span)
    num_joints = size(points_to_smoooth, 1);
    num_axis = size(points_to_smoooth, 3);
    smoothed_pts = nan(size(points_to_smoooth));
    for joint=1:num_joints
        for axis=1:num_axis
            pt_axis_i = squeeze(points_to_smoooth(joint, :, axis));
            smoothd_axis = smooth(pt_axis_i, span);
            smoothed_pts(joint, :, axis) = smoothd_axis;
        end
    end
end