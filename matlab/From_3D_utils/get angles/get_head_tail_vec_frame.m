function head_tail_vec = get_head_tail_vec_frame(head_tail_pts)
    head_tail_vec = squeeze(head_tail_pts(1,:)) - squeeze(head_tail_pts(2,:));
    head_tail_vec = head_tail_vec/norm(head_tail_vec);
end