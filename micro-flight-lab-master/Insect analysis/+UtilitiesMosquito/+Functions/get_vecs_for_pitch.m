function vecs=get_vecs_for_pitch(wing_hull_front,wing_hull_back,num_of_vecs)
% Description:
% returns pitch vectors and their starting points
% 
% Required input:
% wing_hull_front - 3D points on leading edge
% wing_hull_back - 3D points on trailing edge
% num_of_vecs - number of pitch angles to measure
%
% Output:
% vecs - pitch vectors and their starting points
    
    %% align the leading edge's primary vector with the x-axis
    wing_hull_front_mean=mean(wing_hull_front);
    wing_hull_front_cm=wing_hull_front-wing_hull_front_mean; %center the data
    [~,~,V]=svd(wing_hull_front_cm,0);
    wing_main=V(:,1);
    roti = vrrotvec2mat(vrrotvec(wing_main,[1,0,0]));
    wing_hull_front_cm_x=wing_hull_front_cm*roti';
    %% apply transformation to trailing edge
    wing_hull_back_cm=bsxfun(@minus,wing_hull_back,wing_hull_front_mean);
    wing_hull_back_cm_x=wing_hull_back_cm*roti';
    %% restrict points to both edges
    min_cor_front = min(wing_hull_front_cm_x);
    min_cor_back = min(wing_hull_back_cm_x);
    max_cor_front = max(wing_hull_front_cm_x);
    max_cor_back = max(wing_hull_back_cm_x);
    max_cor_x=min([max_cor_front(1),max_cor_back(1)]);
    min_cor_x=max([min_cor_front(1),min_cor_back(1)]);

    length_wing_x=max_cor_x-min_cor_x;
    dx=10e-5; %size of point area to mean on
    vecs=nan(num_of_vecs,6);
    
    % create vectors along the span
    for vec_ind=1:num_of_vecs
        cor_x=min_cor_x+length_wing_x/(num_of_vecs-1)*(vec_ind-1);
        back_point=wing_hull_back_cm_x((wing_hull_back_cm_x(:,1)<(cor_x+dx))&(wing_hull_back_cm_x(:,1)>(cor_x-dx)),:);
        front_point=wing_hull_front_cm_x((wing_hull_front_cm_x(:,1)<(cor_x+dx))&(wing_hull_front_cm_x(:,1)>(cor_x-dx)),:);
        if isempty(back_point)||isempty(front_point)
            % get another close point
        else      
            vec_cm=mean(front_point,1)-mean(back_point,1);
            vec_point_cm=mean(back_point,1);
            vec=vec_cm/roti';
            vec_point=vec_point_cm/roti'+wing_hull_front_mean;
            vecs(vec_ind,:)=[vec_point,vec];
        end
    end
end