function [pnts_distances_next_prev, pnts_distances, mean_distances, distances_std] = get_wings_distances(wings_pts_3D)
    num_frames = size(wings_pts_3D, 2);
    num_points = size(wings_pts_3D, 1);
    num_points_per_wing = num_points/2;
    num_wings = 2;
    left_inds = 1:num_points_per_wing; 
    right_inds = num_points_per_wing+1:num_points;
    inds = [left_inds; right_inds];
    for frame=1:num_frames
        for wing = 1:num_wings
            inds_wing = inds(wing, :);
            dist_mat = squareform(pdist(squeeze(wings_pts_3D(inds_wing,frame ,: ))));
            % find distance between each point to the next point
            for pnt = 1:num_points_per_wing
                [next_pnt, prev_pnt] = get_next_prev_pnt(pnt, num_points_per_wing);
                % find the distances
                dist_to_next = dist_mat(pnt,next_pnt);
                dist_to_prev = dist_mat(pnt,prev_pnt);
                pnts_distances(frame, inds(wing, pnt)) = dist_to_next;
                pnts_distances_next_prev(frame, inds(wing, pnt), 1) = dist_to_next;
                pnts_distances_next_prev(frame, inds(wing, pnt), 2) = dist_to_prev;
            end
        end
    end
    mean_distances = mean(pnts_distances);
    distances_std = std(pnts_distances);
end