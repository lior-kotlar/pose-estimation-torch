function preds_3D_derivative = get_3D_derivative(preds_3D)
    % input: array (num_joints, num_frames, 3) 
    num_frames = size(preds_3D, 2);
    num_points = size(preds_3D, 1);
    for frame=2:num_frames
        for pnt=1:num_points
            pnt_prev = squeeze(preds_3D(pnt, frame - 1 , :));
            pnt_curr = squeeze(preds_3D(pnt, frame, :));
            diff_p = norm(pnt_prev - pnt_curr);
            preds_3D_derivative(pnt, frame) = diff_p;
        end
    end
end