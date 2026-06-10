function new_tail_points = make_tail_tight(old_tail_points,body_center,how_much_tail)
% Description:
% cuts tail in order to get a cleaner direction vector for angle
% calculations
% 
% Required input:
% old_tail_points - 3D points of tail cluster hull
% body_center - 3D point of body's center
% how_much_tail - relative size of remeining tail
%
% Output:
% new_tail_points - 3D points of remaining hull
    
    %% align the tail's primary vector with the x-axis
    old_tail_mean=mean(old_tail_points);
    old_tail_points_cm=old_tail_points-old_tail_mean; %center the data
    [~,~,V]=svd(old_tail_points_cm,0);
    tail_main=V(:,1);
    roti = vrrotvec2mat(vrrotvec(tail_main,[1,0,0]));
    old_tail_cen_main=old_tail_points_cm*roti';
    
    %% apply transformation to body center
    body_center=(body_center-old_tail_mean)*roti';    
    %% determine which side is the tip of the tail and cut from the other side
    min_cor = min(old_tail_cen_main);
    max_cor = max(old_tail_cen_main);
    length_tail_x=max_cor(1)-min_cor(1);
    if abs(min_cor(1)-body_center(1))>abs(max_cor(1)-body_center(1))
        good_part=old_tail_cen_main(old_tail_cen_main(:,1)<...
            (min_cor(1)+how_much_tail*length_tail_x),:);
    else
        good_part=old_tail_cen_main(old_tail_cen_main(:,1)>...
            (max_cor(1)-how_much_tail*length_tail_x),:);
    end
    % reverse transformation
    new_tail_points=good_part/roti'+old_tail_mean;
    
    %% in case, remaining hull is not a single component, keep only largest one
%     % copy hull to grid for using binary image tools
%     diffs=abs(old_tail_points(2,:)-old_tail_points(1,:));
%     voxelSize=min(diffs(diffs>0));
%     [grid_mat,bounds]=points2grid(new_tail_points,voxelSize);
%     conncomp = bwconncomp(grid_mat, 26);
%     [~, maxcell] = max(cellfun(@numel, conncomp.PixelIdxList));
%     if length(conncomp.PixelIdxList)>1
%         biggest_piece = zeros(size(grid_mat));
%         biggest_piece(conncomp.PixelIdxList{1, maxcell}) = 1;
%         x_vec=bounds(1,1):voxelSize:(bounds(2,1)+voxelSize);
%         y_vec=bounds(1,2):voxelSize:(bounds(2,2)+voxelSize);
%         z_vec=bounds(1,3):voxelSize:(bounds(2,3)+voxelSize);
%         % turn the grid back to a list of 3D points
%         [xi,yi,zi] = ind2sub(size(biggest_piece), find(biggest_piece));
%         new_tail_points = [x_vec(xi); y_vec(yi); z_vec(zi)]';  
%     end
end