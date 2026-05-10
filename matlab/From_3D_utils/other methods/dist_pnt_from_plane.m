function distance = dist_pnt_from_plane(plane_P, point_3D)
% plane_P = [a, b, c, d] of plane ax+by+cz+d=0
% point_3D = [x, y, z]
xyz = plane_P(1:3);
d = plane_P(4);
distance = point_3D * xyz + d;
end

