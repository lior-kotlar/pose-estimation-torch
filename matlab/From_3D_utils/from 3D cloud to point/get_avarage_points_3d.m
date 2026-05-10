function avarage_points = get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor)
n_frames = size(all_pts3d, 2);
num_joints = size(all_pts3d, 1);
num_candidates = size(all_pts3d, 3);
avarage_points = nan(num_joints,n_frames ,3);
if num_candidates == 1
    avarage_points = squeeze(all_pts3d);
else
    for frame=1:n_frames
        for joint=1:num_joints
            xyz = squeeze(all_pts3d(joint, frame, :, :)); 
            % extract weight
            if rem_outliers
                [~,indexes_to_remove] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
                r2_arr = xyz(~indexes_to_remove,:);
            else
                r2_arr = xyz;
            end
            
            if size(r2_arr,1) == 1
                avarage_xyz = r2_arr;
            else    
                avarage_xyz = mean(r2_arr);
            end
            avarage_points(joint, frame, :) = avarage_xyz;
        end
    end
end
end