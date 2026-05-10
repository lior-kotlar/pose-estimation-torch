function [preds_2D_derivative, envelope_2D] = get_2D_derivative(preds_2D)
    num_frames = size(preds_2D, 1);
    num_cams = size(preds_2D, 2);
    num_points = size(preds_2D, 3);
    for frame=2:num_frames
        for cam=1:num_cams
            for pnt=1:num_points
                pnt_prev = squeeze(preds_2D(frame - 1, cam, pnt, :));
                pnt_curr = squeeze(preds_2D(frame , cam, pnt, :));
                diff_p = norm(pnt_prev - pnt_curr);
                preds_2D_derivative(frame, cam, pnt) = diff_p;
            end
        end
    end
    for pnt=1:num_points
        deriv_all_frames = squeeze(preds_2D_derivative(:, :, pnt)); 
%         second_derive = zeros(size(deriv_all_frames));
%         second_derive(2:end, :) =  abs(diff(deriv_all_frames));
        [pnt_envelope_2D, ~] = envelope(deriv_all_frames, 10, "peak");
        pnt_envelope_2D(1:4, :) = 0;
        pnt_envelope_2D(end-4:end, :) = 0;
        pnt_envelope_2D = abs(pnt_envelope_2D);
        envelope_2D(:, :, pnt) = pnt_envelope_2D;
%         plot(envelope_2D); hold on; plot(deriv_all_frames, '.') 
    end
end