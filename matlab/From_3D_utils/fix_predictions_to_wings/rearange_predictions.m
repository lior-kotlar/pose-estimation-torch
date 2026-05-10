function predictions = rearange_predictions(preds, num_cams)
% preds is an array of size (num_joints, 2, num_cams*num_frames)
n_frames = size(preds, 3)/num_cams;
num_joints = size(preds, 1);
if num_cams == 4
    for frame_ind=1:n_frames
        preds_temp(:,:,frame_ind)=cat(1,preds(:,:,frame_ind),preds(:,:,frame_ind+n_frames),...
            preds(:,:,frame_ind+2*n_frames),preds(:,:,frame_ind+3*n_frames));
    end
elseif num_cams == 3
    for frame_ind=1:n_frames
        preds_temp(:,:,frame_ind)=cat(1,preds(:,:,frame_ind),preds(:,:,frame_ind+n_frames),...
            preds(:,:,frame_ind+2*n_frames));
    end
end

predictions = zeros(n_frames, num_cams, num_joints, 2);
for frame=1:n_frames
    for cam=1:num_cams
        single_pred = preds_temp((num_joints*(cam-1)+1):(num_joints*(cam-1)+num_joints),:,frame);
        predictions(frame, cam, :, :) = squeeze(single_pred) ;
    end
end
end