function wing_vec = get_wing_vec(wing_pts)
    % 3 different options of were to take take the wing angle: 
    num_pts = size(wing_pts, 1);
    %% get a vector from the shoulder to the tip
    ancor_point = wing_pts(num_pts, :);  % shoulder point
    wing_tip_idx = 3;
    wing_tip = wing_pts(wing_tip_idx, :);
    shoulder_to_tip_vec = (wing_tip - ancor_point)/norm((wing_tip - ancor_point));

    %% get a vector from COM to tip
    wing_COM = mean(wing_pts(1:(num_pts - 1), :));
    COM_to_tip_vec = (wing_tip - wing_COM)/norm((wing_tip - wing_COM));

    %% get a vector from pt1 tp pt2 on the leading edge
    point_2 =  squeeze(wing_pts(2, :));
    point_1 = squeeze(wing_pts(1, :));
    leading_edge_vec = (point_2 - point_1)/norm(point_2 - point_1);
    
    %% deside which vector 
    wing_vec = shoulder_to_tip_vec;
end