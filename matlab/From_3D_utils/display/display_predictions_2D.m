function [] = display_predictions_2D(box, predictions, pause_time) 
    num_frames = size(predictions, 1);
    num_cams = size(predictions, 2);
    num_joints = size(predictions, 3);
    figure('Units','normalized','Position',[0,0,0.9,0.9])
    h_sp(1)=subplot(2,2,1);
    h_sp(2)=subplot(2,2,2);
    h_sp(3)=subplot(2,2,3);
    h_sp(4)=subplot(2,2,4);
    hold(h_sp,'on')
    for cam_ind=1:num_cams
        image = box(:, :, :, cam_ind, 1);
        imshos(cam_ind)=imshow(image,...
            'Parent',h_sp(cam_ind),'Border','tight');
    end
    scats=[];
    texts=[];
    
    for frameInd=1:1:num_frames
        delete(texts)
        delete(scats)
        for cam_ind=1:num_cams
            image = box(:, :, :, cam_ind, frameInd);
            imshos(cam_ind).CData=image;
            this_preds = squeeze(predictions(frameInd, cam_ind, :, :));
            x = this_preds(:,1);
            y = this_preds(:,2);
            scats(cam_ind)=scatter(h_sp(cam_ind),x, y, 44, hsv(num_joints),'LineWidth',3);
            data = ["cam ind = " , string(cam_ind), "frame = ", string(frameInd)]; 
            texts(cam_ind,:) = text(h_sp(cam_ind), 0 ,40 , data,'Color', 'W');    
        end
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
    end
end