function head_tail_vec = get_head_tail_vec_all(points_3D)
    head_pts = squeeze(points_3D(1,:,:));
    tail_points = squeeze(points_3D(2,:,:));
    head_tail_vec = head_pts - tail_points;
    head_tail_vec = normr(head_tail_vec);
end