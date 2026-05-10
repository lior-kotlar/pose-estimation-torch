function [all_planes_dists, all_4_planes, all_2_planes] = get_all_planes(preds_3D)
    num_joints=size(preds_3D,1) - 4;
    left_inds = 1:num_joints/2; 
    right_inds = (num_joints/2+1:num_joints); 
    wings_joints_inds = (num_joints+1:num_joints+2);
    head_tail_inds = (num_joints+3:num_joints+4);
    upper_plane_pnts = [1,2,3,4];
    lower_plane_pnts = [4,5,6,7];
    num_frames = size(preds_3D,2); 
    all_planes_dists = zeros(num_frames, 6);
    all_4_planes = zeros(4, num_frames, 4);
    all_2_planes = zeros(2, num_frames, 4);
    for frame_ind=1:num_frames

        up_left_pnts = squeeze(preds_3D(left_inds(upper_plane_pnts), frame_ind, :));
        P1 = get_plane_params(up_left_pnts);
        dist_upl = mean_dist_pnts_from_plane(P1, up_left_pnts);
        all_planes_dists(frame_ind, 1) = dist_upl;
        all_4_planes(1, frame_ind, :) = P1;

        down_left_pnts = squeeze(preds_3D(left_inds(lower_plane_pnts), frame_ind, :));
        P2 = get_plane_params(down_left_pnts);
        dist_dnl = mean_dist_pnts_from_plane(P2, down_left_pnts);
        all_planes_dists(frame_ind, 3) = dist_dnl;
        all_4_planes(2, frame_ind, :) = P2;

        up_right_pnts = squeeze(preds_3D(right_inds(upper_plane_pnts), frame_ind, :));
        P3 = get_plane_params(up_right_pnts);
        dist_upr = mean_dist_pnts_from_plane(P3, up_right_pnts);
        all_planes_dists(frame_ind, 2) = dist_upr;
        all_4_planes(3, frame_ind, :) = P3;

        down_right_pnts = squeeze(preds_3D(right_inds(lower_plane_pnts), frame_ind, :));
        P4 = get_plane_params(down_right_pnts);
        dist_dnr = mean_dist_pnts_from_plane(P4, down_right_pnts);
        all_planes_dists(frame_ind, 4) = dist_dnr;
        all_4_planes(4, frame_ind, :) = P4;
        
        all_pnts_right = squeeze(preds_3D(right_inds(1:end-1), frame_ind, :));
        P5 = get_plane_params(all_pnts_right);
        dist_r = mean_dist_pnts_from_plane(P5, all_pnts_right);
        all_planes_dists(frame_ind, 5) = dist_r;
        all_2_planes(1, frame_ind, :) = P5;

        all_pnts_left = squeeze(preds_3D(left_inds(1:end-1), frame_ind, :));
        P6 = get_plane_params(all_pnts_left);
        dist_l = mean_dist_pnts_from_plane(P6, all_pnts_left);
        all_planes_dists(frame_ind, 6) = dist_l;
        all_2_planes(2, frame_ind, :) = P6;
    end
end