function display_pnts_1_vs_pnts_2_3D(pnts_1, pnts_2, pause_time)
    x=1;y=2;z=3;
    num_joints = min(size(pnts_1,1) ,size(pnts_2,1));
    num_frames = min(size(pnts_1,2) ,size(pnts_2,2));
    pnts_1 = pnts_1(1:num_joints, 1:num_frames, :);
    pnts_2 = pnts_2(1:num_joints, 1:num_frames, :);
    body_pts=true;
    if num_joints == 14 body_pts=false; end
    if body_pts num_joints=size(pnts_1,1) - 2; end
    left_inds = 1:num_joints/2; 
    right_inds = (num_joints/2+1:num_joints); 
    head_tail_inds = (num_joints+1:num_joints+2);



    num_frames = size(pnts_1,2); 
    figure;
    % imshow(randi([0,1],[191, 4*191]));
    hold on;
    axis equal; 
    box on ; 
    grid on;
    view(3); 
    rotate3d on

    x=1;y=2;z=3;
    max_x = max(pnts_1(:, :, x), [], 'all');
    min_x = min(pnts_1(:, :, x), [], 'all');
    max_y = max(pnts_1(:, :, y), [], 'all');
    min_y = min(pnts_1(:, :, y), [], 'all');
    max_z = max(pnts_1(:, :, z), [], 'all');
    min_z = min(pnts_1(:, :, z), [], 'all');

    xlim1=[min_x, max_x];
    ylim1=[min_y, max_y];
    zlim1=[min_z, max_z];
    scale_box = 1.1;

    xlim(scale_box*(xlim1-mean(xlim1))+mean(xlim1))
    ylim(scale_box*(ylim1-mean(ylim1))+mean(ylim1))
    zlim(scale_box*(zlim1-mean(zlim1))+mean(zlim1))
    p = [];
    % display_predictions_2D(box, predictions, 0);
    for frame_ind=1:num_frames
        % draw fly points 3D
        if body_pts indexes = [8, 16]; right_inds = (1:7); left_inds = (9:15);  end

        p(1) = plot3(pnts_1(right_inds,frame_ind,x),pnts_1(right_inds,frame_ind,y),pnts_1(right_inds,frame_ind,z),'o-g');
        p(2) = plot3(pnts_1(left_inds,frame_ind,x),pnts_1(left_inds,frame_ind,y),pnts_1(left_inds,frame_ind,z),'o-r');
        p(3) = plot3(pnts_2(right_inds,frame_ind,x),pnts_2(right_inds,frame_ind,y),pnts_2(right_inds,frame_ind,z),'+--g');
        p(4) = plot3(pnts_2(left_inds,frame_ind,x),pnts_2(left_inds,frame_ind,y),pnts_2(left_inds,frame_ind,z),'+--r');
 
        if body_pts
            p(5) = plot3(pnts_1(indexes,frame_ind,x),pnts_1(indexes,frame_ind,y),pnts_1(indexes,frame_ind,z),'o-b');
            p(6) = plot3(pnts_1(head_tail_inds,frame_ind,x),pnts_1(head_tail_inds,frame_ind,y),pnts_1(head_tail_inds,frame_ind,z),'o-b');
            p(7) = plot3(pnts_2(indexes,frame_ind,x),pnts_2(indexes,frame_ind,y),pnts_2(indexes,frame_ind,z),'+--b');
            p(8) = plot3(pnts_2(head_tail_inds,frame_ind,x),pnts_2(head_tail_inds,frame_ind,y),pnts_2(head_tail_inds,frame_ind,z),'+--b');
        end

        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
        delete(p);
%         set(p,'Visible','off')
    end