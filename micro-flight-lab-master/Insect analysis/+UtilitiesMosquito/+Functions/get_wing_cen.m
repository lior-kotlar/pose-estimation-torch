function cut_wing_points = get_wing_cen(old_wing_points,body_center)
% Description:
% returns hull without primary axis edges
% 
% Required input:
% old_wing_points - 3D points
%
% Output:
% cut_wing_points - 3D points of remaining hull
    
    %% align the leading edge's primary vector with the x-axis
    old_wing_points_mean=mean(old_wing_points);
    old_wing_points_cm=old_wing_points-old_wing_points_mean; %center the data
    [~,~,V]=svd(old_wing_points_cm,0);
    if dot(mean(old_wing_points)-body_center,V(:,1))<0
        V(:,1)=-V(:,1);
    end
    wing_main=V(:,1);
    roti = vrrotvec2mat(vrrotvec(wing_main,[1,0,0]));
    old_wing_cen_main=old_wing_points_cm*roti';
    %% cut peripherals
    min_cor = min(old_wing_cen_main);
    max_cor = max(old_wing_cen_main);
    length_wing_x=max_cor(1)-min_cor(1);
    
    wing_cut=0.25;
    new_wing_points=old_wing_cen_main((old_wing_cen_main(:,1)>(min_cor(1)+wing_cut*length_wing_x))&...
        (old_wing_cen_main(:,1)<(max_cor(1)-1.0*wing_cut*length_wing_x)),:);
    cut_wing_points=new_wing_points/roti'+old_wing_points_mean;
end