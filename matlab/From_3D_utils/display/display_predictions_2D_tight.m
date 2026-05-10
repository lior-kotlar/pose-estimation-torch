function [] = display_predictions_2D_tight(box, predictions, pause_time) 
    num_frames = size(predictions, 1);
    num_cams = size(predictions, 2);
    num_joints = size(predictions, 3);
    clowd = false;
    if num_joints > 50  % clowd
        clowd = true;
    end

    path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\output_2D.mp4";
    out = VideoWriter(path,'MPEG-4') ;
    out.Quality   = 100 ; % lower quality also means smaller file size
    out.FrameRate = 15 ; % can change that
    open(out) ; 

    fig = figure('Units','normalized','Position',[0,0,0.9,0.9]);
    scats=[];
    texts=[];
    for frameInd=1:num_frames
        t = tiledlayout(2,2);
        t.TileSpacing = 'compact';
        t.Padding = 'compact';
        delete(texts)
        delete(scats)
        for cam_ind=1:num_cams
            nexttile(t);
            image = box(:, :, :, cam_ind, frameInd);
            imshow(image);
            this_preds = squeeze(predictions(frameInd, cam_ind, :, :));
            x = this_preds(:,1);
            y = this_preds(:,2);
            hold on
            if clowd
%                 scatter(x, y, 44, "yellow",'LineWidth',0.1);
                scatter(x(1:num_joints/2), y(1:num_joints/2), 'yellow','*');
                scatter(x(num_joints/2+1:end), y(num_joints/2+1:end), 'blue','*');
            else
                scatter(x, y, 44, hsv(num_joints),'LineWidth',3);
            end
            data = ["cam ind = " , string(cam_ind), "frame = ", string(frameInd)]; 
            hold on
            text(0 ,40 , data,'Color', 'W');    
        end

        frame = getframe(fig);
        writeVideo(out, frame);

        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
    end
end