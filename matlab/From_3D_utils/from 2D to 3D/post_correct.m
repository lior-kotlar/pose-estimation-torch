function preds_3D_corr = post_correct(preds_3D, all_pts3d)
    num_frames = size(preds_3D, 2);
    num_points = size(preds_3D, 1);
    num_wings_pnts = num_points - 4;
    num_points_per_wing = num_wings_pnts/2;
    num_wings = 2;
    left_inds = 1:num_points_per_wing; 
    right_inds = num_points_per_wing+1:num_wings_pnts;
    inds = [left_inds; right_inds];
    angles = find_angles_between_points(preds_3D);
    wings_pnts = preds_3D(1:num_wings_pnts, :, :);
    [pnts_distances_next_prev, pnts_distances, mean_distances, distances_std] = get_wings_distances(wings_pnts);
   preds_3D_corr = preds_3D;
   plot(pnts_distances(:, 10)); hold on; plot(pnts_distances(:, 9));
   hold on; plot(pnts_distances(:, 8));
   hold on; plot(pnts_distances(:, 11));
   C = 1;
   hold on; yline(mean_distances(10) + C*distances_std(10));
   hold on; yline(mean_distances(9) + C*distances_std(9));

    %% find all problematic frames and points and fix them
    for frame=1:num_frames
        for wing=1:num_wings
            for pnt=1:num_points_per_wing
                % find if point is problematic
                [next_pnt, prev_pnt] = get_next_prev_pnt(pnt, num_points_per_wing);
                dist_next = pnts_distances_next_prev(frame, pnt, 1);
                dist_prev = pnts_distances_next_prev(frame, pnt, 2);
                mean_next = mean_distances(inds(wing, next_pnt));
                mean_prev = mean_distances(inds(wing, prev_pnt));
                dist_mean_next = abs(mean_next - dist_next);
                dist_mean_prev = abs(mean_prev - dist_prev);
                std_next = distances_std(inds(wing, next_pnt));
                std_prev = distances_std(inds(wing, prev_pnt));
                
                if dist_mean_prev > std_prev && dist_mean_next > std_next
                    % the distances from the points to it's next and
                    % previous are both 'bad'
                    pnt_ind = inds(wing, pnt);
                    next_pnt_3D = squeeze(preds_3D(inds(wing, next_pnt), frame, :));
                    prev_pnt_3D = squeeze(preds_3D(inds(wing, prev_pnt), frame, :));
                    num_candiadates = size(all_pts3d, 3);
                    cands_distances = zeros(num_candiadates, 3);
                    for cand=1:num_candiadates
                        cand_pnt = squeeze(all_pts3d(pnt_ind, frame, cand, :));
                        dist_next_cand = abs(norm(cand_pnt - next_pnt_3D) - mean_next);
                        dist_prev_cand = abs(norm(cand_pnt - prev_pnt_3D) - mean_prev);
                        cands_distances(cand, 1) = dist_prev_cand + dist_next_cand;
                        cands_distances(cand, 2) = dist_next_cand;
                        cands_distances(cand, 3) = dist_prev_cand;
                    end
                   [B,I] = sort(cands_distances, 1);
                   best_candidate = I(1, 3);
                   preds_3D_corr(pnt_ind, frame, :) = squeeze(all_pts3d(pnt_ind, frame, best_candidate, :));
                end
            end
        end
    end

end