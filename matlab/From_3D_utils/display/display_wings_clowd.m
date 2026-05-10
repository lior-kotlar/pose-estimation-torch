function display_wings_clowd(preds_3D_smoothed ,preds_3D_clowd, pause_time)
    x=1;y=2;z=3;
    num_frames = size(preds_3D_clowd,1); 

    num_joints=size(preds_3D_smoothed,1) - 4;
    left_inds = 1:num_joints/2; 
    right_inds = (num_joints/2+1:num_joints); 

%     left_inds = left_inds(1:end-1);
%     right_inds = right_inds(1:end-1);

    wings_joints_inds = (num_joints+1:num_joints+2);
    head_tail_inds = (num_joints+3:num_joints+4);

    fig = figure();
    hold on;
    axis equal; 
    box on ; 
    grid on;
    view(3); 
    rotate3d on
    all_pnts = cat(2, preds_3D_clowd, permute(preds_3D_smoothed, [2,1,3]));
    max_x = max(all_pnts(:, :, x), [], 'all');
    min_x = min(all_pnts(:, :, x), [], 'all');
    max_y = max(all_pnts(:, :, y), [], 'all');
    min_y = min(all_pnts(:, :, y), [], 'all');
    max_z = max(all_pnts(:, :, z), [], 'all');
    min_z = min(all_pnts(:, :, z), [], 'all');

    xlim1=[min_x, max_x];
    ylim1=[min_y, max_y];
    zlim1=[min_z, max_z];
    scale_box = 1.1;

    xlim(scale_box*(xlim1-mean(xlim1))+mean(xlim1))
    ylim(scale_box*(ylim1-mean(ylim1))+mean(ylim1))
    zlim(scale_box*(zlim1-mean(zlim1))+mean(zlim1))
    left = (1:size(preds_3D_clowd,2)/2);
    right = (size(preds_3D_clowd,2)/2 + 1:size(preds_3D_clowd,2));

    path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils\display\output_3D_black.mp4";
    out = VideoWriter(path,'MPEG-4') ;
    out.Quality   = 100 ; % lower quality also means smaller file size
    out.FrameRate = 15 ; % can change that
    open(out) ; 


    points_to_display = preds_3D_smoothed;
    p = [];
    for frame_ind=1:num_frames
        %% display clowd pnts

        p(1) = plot3(preds_3D_clowd(frame_ind, left, x),preds_3D_clowd(frame_ind, left, y),preds_3D_clowd(frame_ind, left, z),'bo');
        p(2) = plot3(preds_3D_clowd(frame_ind, right, x),preds_3D_clowd(frame_ind, right, y),preds_3D_clowd(frame_ind, right, z),'ro');

        %% display oroginial predicted points
%         hold on;
%         p(3) = plot3(points_to_display(right_inds,frame_ind,x),points_to_display(right_inds,frame_ind,y),points_to_display(right_inds,frame_ind,z),'o-g');
%         hold on;
%         p(4) = plot3(points_to_display(left_inds,frame_ind,x),points_to_display(left_inds,frame_ind,y),points_to_display(left_inds,frame_ind,z),'o-r');
%         hold on;
        p(3) = plot3(points_to_display(head_tail_inds,frame_ind,x),points_to_display(head_tail_inds,frame_ind,y),points_to_display(head_tail_inds,frame_ind,z),'o-b');
        hold on;          
%         p(6) = plot3(points_to_display(wings_joints_inds,frame_ind,x),points_to_display(wings_joints_inds,frame_ind,y),points_to_display(wings_joints_inds,frame_ind,z),'o-b');
        
%         pnts = squeeze(preds_3D_clowd(frame_ind, left, :));
%         k = convhulln(pnts);
%         T(1) = trisurf(k,pnts(:,1),pnts(:,2),pnts(:,3),'FaceColor','black', 'FaceAlpha',1);
% 
%         pnts = squeeze(preds_3D_clowd(frame_ind, right, :));
%         k = convhulln(pnts);
%         T(2) = trisurf(k,pnts(:,1),pnts(:,2),pnts(:,3),'FaceColor','black');

        pnts = squeeze(points_to_display(left_inds, frame_ind, :));
        [X, Y, Z] = get_plane(pnts);
        hold on
        S(1) = surf(X,Y,Z,'FaceAlpha',0.5);
        
        pnts = squeeze(points_to_display(right_inds, frame_ind, :));
        [X, Y, Z] = get_plane(pnts);
        hold on
        S(2) = surf(X,Y,Z,'FaceAlpha',0.5);
% 
%         frame = getframe(fig);
%         writeVideo(out, frame);

        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
        set(p,'Visible','off')
        delete(p);
        delete(S);
    end
end