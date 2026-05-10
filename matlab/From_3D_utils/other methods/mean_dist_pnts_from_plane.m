function mean_dist = mean_dist_pnts_from_plane(plane_P, pnts)
% plane_P = [a, b, c, d] of plane ax+by+cz+d=0
% point_3D = [x, y, z]
xyz = plane_P(1:3)';
d = plane_P(4);
distance = abs(pnts * xyz + d);
mean_dist = mean(distance);
end


