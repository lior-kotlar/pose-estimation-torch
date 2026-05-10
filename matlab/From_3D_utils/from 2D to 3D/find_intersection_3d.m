function intersection_pt = find_intersection_3d(start_2_lines, end_2_lines)
p1 = start_2_lines(1, :);
p2 = end_2_lines(1,:);
p3 = start_2_lines(2, :);
p4 = end_2_lines(2,:);

% Calculate the direction vectors of the two lines
v1 = p2 - p1;
v2 = p4 - p3;

v = cross(v1, v2);
v = v/norm(v);
s = dot(v, p1);
t = dot(v, p2);

point_vec_1 = p1 + s*v1;
point_vec_2 = p3 + t*v2;

intersection_pt = (point_vec_2  + point_vec_1)/2;

% column stack
end