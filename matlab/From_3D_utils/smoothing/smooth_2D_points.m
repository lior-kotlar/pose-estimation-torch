function preds_2D_smoothed = smooth_2D_points(pts_to_smooth_2D, framelen, order)
    num_joints=size(pts_to_smooth_2D,3);
    preds_2D_smoothed = nan(size(pts_to_smooth_2D));
    num_dims = size(pts_to_smooth_2D, 4);
    num_cams = size(pts_to_smooth_2D, 2);
    for node_ind=1:num_joints
        for dim_ind=1:num_dims
            for cam=1:num_cams
                % remove outliers 
                tmp_pnts = squeeze(pts_to_smooth_2D(:, cam, node_ind, dim_ind));
%                 [~,rm_inds] = rmoutliers(tmp_pnts,'movmedian',framelen,'ThresholdFactor',ThresholdFactor);
%                 tmp_pnts(rm_inds) = nan;
                % apply smoothing
                y = sgolayfilt(tmp_pnts, order, framelen);
                preds_2D_smoothed(:, cam, node_ind, dim_ind) = y;
            end
        end
    end
end
