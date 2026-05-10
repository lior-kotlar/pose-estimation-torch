function points_3D_sewed = sew_path_using_MSE(all_pts3d)
num_nodes = size(all_pts3d, 1);
num_frames = size(all_pts3d, 2);
for node=1:num_nodes
    first_nodes_pnts = squeeze(all_pts3d(node, 1, :, :));
    % remove outliers
    [~,inlierIndices,~] = pcdenoise(pointCloud(first_nodes_pnts), Threshold=1);
    first_node_pnts = first_nodes_pnts(inlierIndices,:);
    best_estimates(1,:) = mean(first_node_pnts);
    for frame=2:num_frames
        candidate_pts = squeeze(all_pts3d(node, frame, :, :));
        % remove outliers
        [~,inlierIndices,~] = pcdenoise(pointCloud(candidate_pts), Threshold=1);
        candidate_pts = candidate_pts(inlierIndices,:);
        num_candidates = size(candidate_pts, 1);
        errors = zeros(num_candidates, 1);
        for cand=1:num_candidates
            errors(cand) = norm(candidate_pts(cand, :) - best_estimates(frame - 1, :));
        end
        [~, minIndex] = min(errors);
        best_estimates(frame, :) = candidate_pts(minIndex, :);
    end
    points_3D_sewed(node, :, :) = best_estimates;
end
end