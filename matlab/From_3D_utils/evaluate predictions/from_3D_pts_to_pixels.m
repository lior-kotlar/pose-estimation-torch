function from_2d_to_3d_pts = from_3D_pts_to_pixels(pt3d, easyWandData, cropzone)
    allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
    n_frames = size(pt3d, 2);
    num_joints = size(pt3d, 1);
    num_cams = length(allCams.cams_array);
    for frame = 1:n_frames
        for cam_ind=1:num_cams
            for joint=1:num_joints
                joint_pt = allCams.Rotation_Matrix' * squeeze(pt3d(joint, frame, :));
                xy_per_cam_per_joint = dlt_inverse(allCams.cams_array(cam_ind).dlt, joint_pt');
                % flip y
                xy_per_cam_per_joint(2) = 801 - xy_per_cam_per_joint(2);
                x_p = xy_per_cam_per_joint(1); y_p = xy_per_cam_per_joint(2);
                % crop
                x_crop = cropzone(2, cam_ind, frame);
                y_crop = cropzone(1, cam_ind, frame);
                x = x_p - x_crop; 
                y = y_p - y_crop;
                from_2d_to_3d_pts(joint, 1, cam_ind, frame) = x;
                from_2d_to_3d_pts(joint, 2, cam_ind, frame) = y;
            end
        end
    end
    from_2d_to_3d_pts = permute(from_2d_to_3d_pts, [4,3,1,2]);
end