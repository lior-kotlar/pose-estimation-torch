function display_predictions_2D_and_3D(images, pts_3D, pts_2D, pause_time)
    num_frames = size(pts_2D, 1);
    num_cams = size(pts_2D, 2);
    num_joints = size(pts_2D, 3);
    num_wing_pts = num_joints - 2;
    left_inds = 1:num_wing_pts/2; 
    right_inds = (num_wing_pts/2+1:num_wing_pts); 
    head_tail_inds = (num_wing_pts+1:num_wing_pts+2);
    x=1;y=2;z=3;
    max_x = max(pts_3D(:, :, x), [], 'all');
    min_x = min(pts_3D(:, :, x), [], 'all');
    max_y = max(pts_3D(:, :, y), [], 'all');
    min_y = min(pts_3D(:, :, y), [], 'all');
    max_z = max(pts_3D(:, :, z), [], 'all');
    min_z = min(pts_3D(:, :, z), [], 'all');
    xlim1=[min_x, max_x];
    ylim1=[min_y, max_y];
    zlim1=[min_z, max_z];
    scale_box = 1.5;
  

    figure('Units','normalized','Position',[0,0,0.9,0.9])
    h_sp(1)=subplot(2,3,1);
    h_sp(2)=subplot(2,3,2);
    h_sp(3)=subplot(2,3,3);
    h_sp(4)=subplot(2,3,4);
    h_sp(5)=subplot(2,3,[5,6]);
    hold(h_sp,'on')

    axes(h_sp(5));
    axis equal; 
    box on ; 
    grid on;
    view(3); 
    rotate3d on
    xlim(scale_box*(xlim1-mean(xlim1))+mean(xlim1))
    ylim(scale_box*(ylim1-mean(ylim1))+mean(ylim1))
    zlim(scale_box*(zlim1-mean(zlim1))+mean(zlim1))

    for cam_ind=1:num_cams
        image = images(:, :, :, cam_ind, 1);
        imshos(cam_ind)=imshow(image,...
            'Parent',h_sp(cam_ind),'Border','tight');
    end
    scats=[];
    texts=[];
    p = [];
    for frame_ind=1:1:num_frames
        delete(texts)
        delete(scats)
%         delete(p)
        for cam_ind=1:num_cams
            image = images(:, :, :, cam_ind, frame_ind);
            imshos(cam_ind).CData=image;
            this_preds = squeeze(pts_2D(frame_ind, cam_ind, :, :));
            xs = this_preds(:,1);
            ys = this_preds(:,2);
            scats(cam_ind)=scatter(h_sp(cam_ind),xs, ys, 44, hsv(num_joints),'LineWidth',3);
            data = ["cam ind = " , string(cam_ind), "frame = ", string(frame_ind)]; 
            texts(cam_ind,:) = text(h_sp(cam_ind), 0 ,40 , data,'Color', 'W');    
        end
     
        axes(h_sp(5));
        set(p,'Visible','off')
        p(1) = plot3(pts_3D(left_inds,frame_ind,x),pts_3D(left_inds,frame_ind,y),pts_3D(left_inds,frame_ind,z),'o-r');
        p(2) = plot3(pts_3D(right_inds,frame_ind,x),pts_3D(right_inds,frame_ind,y),pts_3D(right_inds,frame_ind,z),'o-g');
        p(3) = plot3(pts_3D(head_tail_inds,frame_ind,x),pts_3D(head_tail_inds,frame_ind,y),pts_3D(head_tail_inds,frame_ind,z),'o-b');
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
        
    end
end