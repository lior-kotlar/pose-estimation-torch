function distance = get_distance_predictions_labels(labels, predictions)
num_joints = size(predictions, 1);
num_cams = size(predictions, 3);
n_frames = size(predictions, 4);
distance = nan(num_joints, num_cams, n_frames);
for frame=1:n_frames
    for cam=1:num_cams
        for point=1:num_joints
            p1 = labels(point, :, cam, frame);
            p2 = predictions(point, :, cam, frame);
            d = norm(p1 - p2); 
            distance(point, cam, frame) = d;
        end
    end
end
sz = size(distance);
dis_cam1 = distance(:,1,:); dis_cam2 = distance(:,2,:); 
dis_cam3 = distance(:,3,:); dis_cam4 = distance(:,4,:);
distance = squeeze(cat(3, dis_cam1, dis_cam2, dis_cam3, dis_cam4)); 
end