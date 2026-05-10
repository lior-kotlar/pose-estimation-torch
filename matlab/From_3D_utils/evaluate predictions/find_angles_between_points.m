function angles = find_angles_between_points(preds_3D)
    num_frames = size(preds_3D, 2);
    num_joints = size(preds_3D, 1);
    num_body_parts = 4;
    num_wings = 2;
    num_wings_pts = num_joints - num_body_parts;
    pnt_per_wing = num_wings_pts/2;
    left_inds = 1:pnt_per_wing; 
    right_inds = (pnt_per_wing+1:num_wings_pts); 
    inds = [left_inds; right_inds];
    for frame=1:num_frames
        for wing = 1:num_wings
            for pnt = 1:pnt_per_wing
                next_pnt = mod(pnt + 1, pnt_per_wing);
                prev_pnt = mod(pnt - 1, pnt_per_wing);
                if prev_pnt == 0 prev_pnt = pnt_per_wing; end 
                if next_pnt == 0 next_pnt = pnt_per_wing; end
                prev_ind = inds(wing, prev_pnt);
                ind = inds(wing, pnt);
                next_ind = inds(wing, next_pnt);
                p1 = squeeze(preds_3D(prev_ind, frame, :));
                p2 = squeeze(preds_3D(ind, frame, :));
                p3 = squeeze(preds_3D(next_ind, frame, :));

                vec1 = p2 - p1;
                vec2 = p3 - p2;
                cos_alpha = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
                alpha = acos(cos_alpha);
                alpha_deg = 180 - rad2deg(alpha);
                angles(frame, wing, pnt) = alpha_deg;
            end
        end
    end
end