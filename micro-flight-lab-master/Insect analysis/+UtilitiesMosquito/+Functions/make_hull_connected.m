function new_hull_points = make_hull_connected(old_hull_points)
% Description:
% returns largest connected component after closing image with small filter
% 
% Required input:
% old_hull_points - 3D points of wing cluster hull
%
% Output:
% new_hull_points - 3D points of remaining hull
 
    %% in case, remaining hull is not a single component, keep only largest one
    % copy hull to grid for using binary image tools
%     diffs=abs(old_hull_points(2,:)-old_hull_points(1,:));
%     voxelSize=min(diffs(diffs>0));
%     [grid_mat,bounds]=points2grid(old_hull_points,voxelSize);
%     x_vec=bounds(1,1):voxelSize:(bounds(2,1)+voxelSize);
%     y_vec=bounds(1,2):voxelSize:(bounds(2,2)+voxelSize);
%     z_vec=bounds(1,3):voxelSize:(bounds(2,3)+voxelSize);
%     
%     grid_mat=imclose(grid_mat,strel('sphere',5)); % fill small holes
%     conncomp = bwconncomp(grid_mat, 26);
%     [~, maxcell] = max(cellfun(@numel, conncomp.PixelIdxList));
%     if length(conncomp.PixelIdxList)>1
%         biggest_piece = zeros(size(grid_mat));
%         biggest_piece(conncomp.PixelIdxList{1, maxcell}) = 1;
%         % turn the grid back to a list of 3D points
%         [xi,yi,zi] = ind2sub(size(biggest_piece), find(biggest_piece));
%         new_hull_points = [x_vec(xi); y_vec(yi); z_vec(zi)]';
%     else
%         % turn the grid back to a list of 3D points
%         [xi,yi,zi] = ind2sub(size(grid_mat), find(grid_mat));
%         new_hull_points = [x_vec(xi); y_vec(yi); z_vec(zi)]';
%     end
    
    T = clusterdata(old_hull_points,'Criterion','distance','cutoff',5e-4);
    num_clusts=max(T)-min(T)+1;
%     color_matrix=lines(num_clusts);
%     figure;
%     scatter3(old_hull_points(:,1), old_hull_points(:,2), old_hull_points(:,3), 36, color_matrix(T,:), 'Marker','.')
%     axis equal vis3d;
    cluster_sizes=sum(T==1:num_clusts,1);
    [~,big_clust_ind]=max(cluster_sizes);
    new_hull_points=old_hull_points(T==big_clust_ind,:);
end