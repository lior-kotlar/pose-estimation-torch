function [points_3D_ground_truth, points_2D] = get_3d_points_ground_truth(cropzone, easyWandData)

    allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
    centers=allCams.all_centers_cam';
    % camera 
    cam_inds = [1, 4];
    crop = cropzone(:, [1, 4], 12);
    centers = centers(cam_inds, :);
    % frame 12
    %% cam 1
    % P8
    x = 71; y = 96;
    points_2D(1, 1, 1) = x; points_2D(1, 1, 2) = y;
    
    % P9
    x = 47; y = 105;
    points_2D(1, 2, 1) = x; points_2D(1, 2, 2) = y;

    % P10
    x = 43; y = 114;
    points_2D(1, 3, 1) = x; points_2D(1, 3, 2) = y;
    
    % P11
    x = 46; y = 118;
    points_2D(1, 4, 1) = x; points_2D(1, 4, 2) = y;
    
    % P12
    x = 63; y = 117;
    points_2D(1, 5, 1) = x; points_2D(1, 5, 2) = y;

    % P13
    x = 81; y = 104;
    points_2D(1, 6, 1) = x; points_2D(1, 6, 2) = y;
    
    % P14
    x = 65; y = 108;
    points_2D(1, 7, 1) = x; points_2D(1, 7, 2) = y;
    
    %% Cam 4
    
    % P8
    x = 90; y = 76;
    points_2D(2, 1, 1) = x; points_2D(2, 1, 2) = y;
    
    % P9
    x = 67; y = 66;
    points_2D(2, 2, 1) = x; points_2D(2, 2, 2) = y;
    
    % P10
    x = 56; y = 67;
    points_2D(2, 3, 1) = x; points_2D(2, 3, 2) = y;
    
    % P11
    x = 53; y = 71;
    points_2D(2, 4, 1) = x; points_2D(2, 4, 2) = y;
    
    % P12
    x = 64; y = 87;
    points_2D(2, 5, 1) = x; points_2D(2, 5, 2) = y;
    
    % P13
    x = 87; y = 88;
    points_2D(2, 6, 1) = x; points_2D(2, 6, 2) = y;
    
    % P14
    x = 75; y = 78;
    points_2D(2, 7, 1) = x; points_2D(2, 7, 2) = y;

    
    for node_ind=1:7
        for cam=1:2
            x = double(crop(2,cam))+squeeze(points_2D(cam, node_ind, 1));
            y = double(crop(1,cam))+squeeze(points_2D(cam, node_ind, 2));
            pose = [x; 801 - y; 1];
            PB(cam, :) = allCams.cams_array(cam_inds(cam)).invDLT*pose;
        end
        PB_norm = PB(:,1:3)./PB(:,4);
        PA = centers;
        [pt_3D, error] = HullReconstruction.Functions.lineIntersect3D(PA,PB_norm);
        pt_3D = pt_3D * allCams.Rotation_Matrix';
        points_3D_ground_truth(node_ind, :) = pt_3D;
    end

end