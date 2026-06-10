function new_wing_points = make_thin_wing(old_wing_points,body_center)
% Description:
% cuts wings in order to get a cleaner direction vector for angle
% calculations
% 
% Required input:
% old_wing_points - 3D points of wing cluster hull
% body_center - 3D point of body's center
%
% Output:
% new_wing_points - 3D points of remaining hull
    
%     %% align the wings's primary vector with the x-axis
%     old_wing_points_cen=mean(old_wing_points);
%     old_wing_points_cm=old_wing_points-old_wing_points_cen; %center the data
%     [~,~,V]=svd(old_wing_points_cm,0);
%     wing_main=V(:,1);
%     roti = vrrotvec2mat(vrrotvec(wing_main,[1,0,0]));
%     old_wing_cen_main=old_wing_points_cm*roti';
%     %% apply transformation to body center
%     body_center=(body_center-old_wing_points_cen)*roti';
%     %% determine which side is the tip of the wing and cut from the other side
%     min_cor = min(old_wing_cen_main);
%     max_cor = max(old_wing_cen_main);
%     middle_of_wing_x=(max_cor(1)+min_cor(1))/2;
%     if abs(min_cor(1)-body_center(1))>abs(max_cor(1)-body_center(1))
%         good_half=old_wing_cen_main(old_wing_cen_main(:,1)<middle_of_wing_x,:);
%     else
%         good_half=old_wing_cen_main(old_wing_cen_main(:,1)>middle_of_wing_x,:);
%     end
%     %% align good_half's primary vector with the x-axis
%     good_half_cm=good_half-mean(good_half); %center the data
%     [~,~,V]=svd(good_half_cm,0);
%     good_half_cm_main=V(:,1);
%     good_half_cm_second=V(:,2);
%     roti2 = vrrotvec2mat(vrrotvec(good_half_cm_main,[1,0,0]));
%     roti3 = vrrotvec2mat(vrrotvec(roti2*good_half_cm_second,[0,1,0]));
%     old_wing_cen_main_from_good_half=(roti3*roti2*old_wing_cen_main')';
%     
%     newpoints  = old_wing_cen_main_from_good_half(:,[1,3]);
%     min_cor = min(newpoints);
%     max_cor = max(newpoints);
%     length_wing_x=max_cor(1)-min_cor(1);
%     wing_inc=length_wing_x/40;
%     middle_of_wing_x=(max_cor(1)+min_cor(1))/2;
%     zeds=newpoints(newpoints(:,1)>middle_of_wing_x,2);
%     width_half=max(zeds)-min(zeds);
%     for i=1:20
%         zeds=newpoints(newpoints(:,1)>middle_of_wing_x-wing_inc*i,2);
%         width=max(zeds)-min(zeds);
%         good_wing=old_wing_cen_main_from_good_half(...
%                 old_wing_cen_main_from_good_half(:,1)>middle_of_wing_x-wing_inc*(i-1),:);
%         if width>1.2*width_half
%             break
%         end
%     end
%     % reverse transformation
%     new_wing_points=(((good_wing/roti3')/roti2')/roti')+old_wing_points_cen;

%% in case, remaining hull is not a single component, keep only largest one
    % copy hull to grid for using binary image tools
    diffs=abs(old_wing_points(2,:)-old_wing_points(1,:));
    voxelSize=min(diffs(diffs>0));
    [grid_mat,bounds]=points2grid(old_wing_points,voxelSize);
    x_vec=bounds(1,1):voxelSize:(bounds(2,1)+voxelSize);
    y_vec=bounds(1,2):voxelSize:(bounds(2,2)+voxelSize);
    z_vec=bounds(1,3):voxelSize:(bounds(2,3)+voxelSize);
    
    grid_mat=imclose(grid_mat,strel('sphere',5)); % fill small holes
    conncomp = bwconncomp(grid_mat, 26);
    [~, maxcell] = max(cellfun(@numel, conncomp.PixelIdxList));
    if length(conncomp.PixelIdxList)>1
        biggest_piece = zeros(size(grid_mat));
        biggest_piece(conncomp.PixelIdxList{1, maxcell}) = 1;
        % turn the grid back to a list of 3D points
        [xi,yi,zi] = ind2sub(size(biggest_piece), find(biggest_piece));
        new_wing_points = [x_vec(xi); y_vec(yi); z_vec(zi)]';
    else
        % turn the grid back to a list of 3D points
        [xi,yi,zi] = ind2sub(size(grid_mat), find(grid_mat));
        new_wing_points = [x_vec(xi); y_vec(yi); z_vec(zi)]';
    end
end