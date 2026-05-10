function [dist_mean, dist_variance, points_distances]= get_wing_distance_variance(points_3d)
% returns:
% dist_mean: dist_mean[i] = distance from point i to point i+1
% dist_variance: dist_variance[i] = variance of distance from point i to point i+1
% points_distances: points_distances[frame, i] = the distance from point i
% to point i+1 in 'frame'.
all_pts = false;
if size(points_3d, 1) == 16
    head_tail_pts =squeeze(points_3d(15:16, :,:));
    points_3d = points_3d(1:14, :,:);
    all_pts = true;
end

num_points = size(points_3d, 1);
n_frames = size(points_3d, 2);
% points_distances = nan(n_frames, num_points);
for frame=1:n_frames
    if num_points == 2
        p1 = points_3d(1, frame, :);
        p2 = points_3d(2, frame, :);
        points_distances(frame) = norm(squeeze(p1- p2));
    else
        % get distance matrix for each frame
        dist_mat = squareform(pdist(squeeze(points_3d(:,frame ,: ))));
        if any(isnan(dist_mat))
            a=0;
        end
        % get the distances between every 2 consecutive points
        for i= 1:num_points/2 - 1
            points_distances(frame, i) = dist_mat(i,i+1);
            j = i + num_points/2;
            points_distances(frame, j) = dist_mat(j,j+1);
        end
        if all_pts
            head = squeeze(head_tail_pts(1, frame, :));
            tail = squeeze(head_tail_pts(2, frame, :));
            points_distances(frame, num_points/2) = norm(head - tail);
        end
    end
end
dist_variance = nanvar(points_distances, 0, 1);
dist_mean = mean(points_distances);
end