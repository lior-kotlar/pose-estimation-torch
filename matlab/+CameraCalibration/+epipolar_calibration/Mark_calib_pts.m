function [xs,ys]=Mark_calib_pts(sparse_folder_path,easy_wand_path,mov_num,good_frames,varargin)
%{
TO DO:
-MAKE TRUE BLACK LEGS STRONG THRESHOLD
%}
%{
Description:
-----------
takes sparse movie files (triplets), crops the image and creates a data structure
compatible with the NN

Input:
-----
sparse_folder_path - path to sparse movie files.
crop_size (Optional) - size of crop window.
num_time_channels (optional name-value pair) - number of time channels for
    each frame for each camera.
time_jump (optional name-value pair) - delay between frames in the
    num_time_channels set.
numCams (optional name-value pair) - number of cameras (3?)

Output:
-----
saved_file_names - cell array of saved h5 files

Example:
-------
LegTrackerGui(sparse_folder_path,easy_wand_path)
%}
    %% parse inputs and initialize variables
    inp_parser = inputParser;
    addRequired(inp_parser,'sparse_folder_path');
    addRequired(inp_parser,'easy_wand_path');
    addRequired(inp_parser,'mov_num');
    addRequired(inp_parser,'good_frames');
    addParameter(inp_parser,'num_marks',6);
    parse(inp_parser,sparse_folder_path,easy_wand_path,mov_num,good_frames,varargin{:});
    % load easywand data and create the all_cameras_class
    easy_wand_data=load(easy_wand_path);
    all_cams=HullReconstruction.Classes.all_cameras_class(easy_wand_data.easyWandData);
    % load sparse movie files and load backgrounds to cameras
    file_names=dir(fullfile(sparse_folder_path,['mov',mov_num,'_cam*.mat']));
    file_names={file_names.name};
    
%     switch to 5234
%     file_names=circshift(file_names,1);
    % clip to 234
%     file_names(4)=[];
    
    start_frame=good_frames(1);
    n_frames=length(good_frames);

    padding=10;
    mf=cellfun(@(x) matfile(fullfile(sparse_folder_path,x)),file_names,...
            'UniformOutput',false);
%     all_movie_lengths=cellfun(@(x) size(x,'frames',1),mf); 
    all_meta_data=cellfun(@(x) x.metaData,mf);
%     frame_offsets=max([all_meta_data.startFrame])-[all_meta_data.startFrame];
%     n_frames_good=min(all_movie_lengths-frame_offsets);
    
    frames=cellfun(@(x) x.frames,mf,'UniformOutput',false);

    % FLIP mirrored CAM 5     
%     for frame_ind=1:length(frames{1})
%         frames{1}(frame_ind).indIm(:,1)=(all_meta_data(1).frameSize(1)+1)-frames{1}(frame_ind).indIm(:,1);
%     end
    
    num_cams=length(frames);
    col_mat=hsv(num_cams);
    
    %% create axes
    handles.mainFig = figure('WindowState','maximized');
    handles.mainLayout = uix.VBox( 'Parent', handles.mainFig );
    handles.sparseMoviesLayout = uix.HBox('Parent', handles.mainLayout, 'Spacing', 3);
    for cam_ind=1:num_cams
        handles.axesCamPanel(cam_ind)=uix.BoxPanel( 'Title', ['Cam',num2str(cam_ind)],...
            'Parent', handles.sparseMoviesLayout ,'TitleColor',col_mat(cam_ind,:));
        handles.hAxes(cam_ind) = axes( 'Parent', handles.axesCamPanel(cam_ind), ...
            'ActivePositionProperty', 'outerposition','Units', 'Normalized', 'Position', [0 0 1 1],...
            'NextPlot','add','Tag',num2str(cam_ind));
        all_cams.cams_array(cam_ind).load_background(all_meta_data(cam_ind).bg);
        
        frame=frames{cam_ind}(start_frame);

%         full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
%         full_im(sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2)))=frame.indIm(:,3);
%         full_im(~bwareafilt(full_im>0,1))=0;

        full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
        lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
        full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3);

        [r,c,v] = find(full_im);
        frame.indIm=[r,c,v];
        
        handles.hImage(cam_ind)=imshow(full_im,[],'parent',handles.hAxes(cam_ind));
        
        max_find_row=double(max(frame.indIm(:,1)));
        min_find_row=double(min(frame.indIm(:,1)));
        max_find_col=double(max(frame.indIm(:,2)));
        min_find_col=double(min(frame.indIm(:,2)));
        
        xlim(handles.hAxes(cam_ind),[min_find_col-padding,max_find_col+padding])
        ylim(handles.hAxes(cam_ind),[min_find_row-padding,max_find_row+padding])

        hold(handles.hAxes(cam_ind),'on')
    end

    % initial marking
    % loop on legs
    for mark_ind=1:inp_parser.Results.num_marks
        [x,y,tags]=CameraCalibration.epipolar_calibration.Mark2d(all_cams,handles.mainFig);
        [~,I]=sort(tags);
        
        xs(:,mark_ind)=x(I);
        ys(:,mark_ind)=y(I);
    end
    
    delete(findobj('Tag','epi_line'))
    
    close(handles.mainFig)
end