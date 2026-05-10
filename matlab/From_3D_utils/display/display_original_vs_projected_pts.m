function display_original_vs_projected_pts(box,original_2D_pts, projected_2D_pts, cameras_used, pause_time)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
    num_frames = size(original_2D_pts, 1);
    num_cams = size(original_2D_pts, 2);
    num_joints = size(original_2D_pts, 3);
    figure('Units','normalized','Position',[0,0,0.9,0.9])
    scats=[];
    texts=[];
    for frameInd=1:num_frames
        t = tiledlayout(2,2);
        t.TileSpacing = 'compact';
        t.Padding = 'compact';
        delete(texts)
        delete(scats)
        cameras_used_wing_1 = squeeze(cameras_used(frameInd, 1, :));
        cameras_used_wing_2 = squeeze(cameras_used(frameInd, 2, :));    
        for cam_ind=1:num_cams
            nexttile(t);
            image = box(:, :, :, cam_ind, frameInd);
            imshow(image);

            proj_preds = squeeze(projected_2D_pts(frameInd, cam_ind, 1:num_joints, :));
            x_proj = proj_preds(:,1);
            y_proj = proj_preds(:,2);

            orig_preds = squeeze(original_2D_pts(frameInd, cam_ind, :, :));
            x_orig = orig_preds(:,1);
            y_orig = orig_preds(:,2);
%             hold on
%             scatter(x_proj(9), y_proj(9), 'Marker', '+', 'LineWidth', 2);
            hold on
            scatter(x_orig, y_orig, 44, hsv(num_joints),'LineWidth',3);
            hold on
            scatter(x_proj, y_proj, 'Marker', '+', 'SizeData', 50, 'CData',  hsv(num_joints), 'LineWidth', 2); 
            
            taken_for_wing_1 = "false";
            if ismember(cam_ind, cameras_used_wing_1)
                taken_for_wing_1 = "true"; 
            end
            
            taken_for_wing_2 = "false";
            if ismember(cam_ind, cameras_used_wing_2)
                taken_for_wing_2 = "true"; 
            end

            data = ["cam ind = " , string(cam_ind), "frame = ", string(frameInd),... 
%                 "red wing", taken_for_wing_1,...
%                 "blue wing", taken_for_wing_2,
                ]; 
            hold on
            text(0 ,40 , data,'Color', 'W');    
            hold on
            for joint=1:num_joints 
                line([x_orig(joint), x_proj(joint)], [y_orig(joint), y_proj(joint)], 'Color','yellow')
            end
        end
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
    end
end