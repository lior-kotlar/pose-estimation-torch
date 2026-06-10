function ds=dist_ln_pt(lns,pts)
% Description:
% calculates distances between lines and points in 2D
% 
% Required input:
% lns - matrix of lines; matrix of num_of_lines*3 (a,b,c) [homogeneous]
% pts - matrix of points; matrix of 3*num_of_pts (x,y,1) [homogeneous]
%
% Output:
% ds - matrix of all distances

    ds=abs(lns*pts)./sqrt(lns(:,1).^2+lns(:,2).^2);
end