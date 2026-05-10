function SparseMovieLoaderV2()
%{
Description:
Creates a gui for playing sparse mat files

- made using the GUILayout toolbox (has to be preinstalled)
%}
    %% create all controls
    %% create axes
    handles.mainFig = figure('CloseRequestFcn',@FigCloseReq);
    handles.mainLayout = uix.VBox( 'Parent', handles.mainFig );
    handles.sparseMoviesLayout = uix.HBox('Parent', handles.mainLayout, 'Spacing', 3);
    for axesInd=1:3
       handles.axesCamPanel(axesInd)=uix.BoxPanel( 'Title', ['Cam',num2str(axesInd)], 'Parent', handles.sparseMoviesLayout );
       handles.hAxes(axesInd) = axes( 'Parent', handles.axesCamPanel(axesInd), ...
        'ActivePositionProperty', 'outerposition','Units', 'Normalized', 'Position', [0 0 1 1]);
        handles.hImage(axesInd)=imshow(rand(100),'parent',handles.hAxes(axesInd));
    end
    %% create buttons
    % Load buttons
    handles.buttonsLayout = uix.HBox( 'Parent', handles.mainLayout );
    handles.loadButtonsLayout = uix.VButtonBox( 'Parent', handles.buttonsLayout, 'Padding', 1,...
        'ButtonSize', [130 35]);
    handles.loadButton=uicontrol( 'Parent', handles.loadButtonsLayout, ...
        'String', 'Load' ,'Callback',@LoadButtonCallback);
    handles.textMovName = uicontrol(handles.loadButtonsLayout,'style','Text',...
        'string','Mov name:');
    % control buttons
    handles.playLayout = uix.VBox( 'Parent', handles.buttonsLayout );
    handles.frameNumbersLayout = uix.HBox( 'Parent', handles.playLayout );
    handles.textStartFrame = uicontrol(handles.frameNumbersLayout,'style','Edit',...
                'string','1','Enable','off');
    handles.textCurrFrame = uicontrol(handles.frameNumbersLayout,'style','Edit',...
        'string','1','Enable','inactive','Callback',@CurrFrameEdit);
    handles.textEndFrame = uicontrol(handles.frameNumbersLayout,'style','Edit',...
        'string','1','Enable','off');
    set( handles.frameNumbersLayout, 'Widths', [-1,-6,-1], 'Spacing', 1 );

    handles.frameSlider = uicontrol( 'Parent', handles.playLayout,'Style', 'slider',...
                'Min',1,'Max',1,'Value',1,...
                'Enable','off');
    addlistener(handles.frameSlider, 'Value', 'PostSet',@ContinuousSlide);

    handles.playButtonsLayout = uix.HButtonBox( 'Parent', handles.playLayout, 'Padding', 5 ,...
        'ButtonSize', [90 60],'Spacing',20);
    handles.firstFrameButton=uicontrol('Parent',handles.playButtonsLayout, ...
        'String','First','Enable','inactive','callback', ...
        @FirstFrameCallback);
    handles.rewindToggle=uicontrol('Style','togglebutton', 'Parent',handles.playButtonsLayout, ...
        'String','Rewind','Enable','inactive','callback', ...
        @RewindPauseCallback);
    handles.playToggle=uicontrol('Style','togglebutton', 'Parent',handles.playButtonsLayout, ...
        'Value',0,'String','Play','Enable','inactive','callback', ...
        @PlayPauseCallback);
    handles.lastFrameButton=uicontrol('Parent',handles.playButtonsLayout, ...
        'String','Last','Enable','inactive','callback', ...
        @LastFrameCallback);
    handles.bgChkbx=uicontrol('Parent',handles.playButtonsLayout,...
        'style','checkbox','string','Add bg',...
        'Enable','off','Value',0);
    handles.textLabelSkip = uicontrol(handles.playButtonsLayout,'style','Text',...
        'string','Skip:');
    handles.textEditSkip = uicontrol(handles.playButtonsLayout,'style','Edit',...
        'string','1','Enable','inactive','Callback',@SkipEdit);
    handles.textCurrMs = uicontrol(handles.playButtonsLayout,'style','Text',...
        'string','time: ms');
    %% set ratios between elements
    set( handles.buttonsLayout , 'Widths', [-1,-6]);
    set( handles.mainLayout, 'Heights', [-6 -1], 'Spacing', 5 );
    
    %% callbacks
    function LoadButtonCallback(~, ~)
       % [fileNames,pathName] = uigetfile('*.mat','Select the sparse file/s',...
        %    'MultiSelect', 'on','C:\Users\noamler\Downloads\all_cut_2018_05_21');
           [fileNames,pathName] = uigetfile('*.mat','Select the sparse file/s',...
            'MultiSelect', 'on','D:\NoamT\180820DarkFlyMagnet\sync');
        if isequal(fileNames,0)
           disp('User selected Cancel')
           return
        end
        if ~iscell(fileNames)
            fileNames={fileNames};
        end
        if length(fileNames)>3
            %%add - error
            warndlg('You can load up to 3 movies at a time!')
            return
        end
        name_split=strsplit(fileNames{1},'_');
        handles.textMovName.String=['Movie Name: ',name_split{1}];

        handles.sparseMatFiles=cell(length(fileNames),1);
        handles.sparseMatFiles=cellfun(@(x) matfile(fullfile(pathName,x)),...
            fileNames,'UniformOutput',false);
        % synchronize all movies and cut excess data
        allMovieLengths=cellfun(@(x) size(x,'frames',1),handles.sparseMatFiles);
        allStartFrames=cellfun(@(x) x.metaData,handles.sparseMatFiles);
        frameOffsets=max([allStartFrames.startFrame])-[allStartFrames.startFrame];
        nFramesGood=min(allMovieLengths-frameOffsets);
        handles.metaData=handles.sparseMatFiles{1}.metaData;
        handles.bgs=zeros([size(handles.metaData.bg),length(fileNames)],...
            'like',handles.metaData.bg);
        if isfield(handles,'sparseMovies')
            handles = rmfield(handles,'sparseMovies');
        end
        arrayfun(@(x) imshow(rand(100),'parent',x),...
            handles.hAxes)
        handles.hImage(axesInd)=imshow(rand(100),'parent',handles.hAxes(axesInd));
        for fileInd=1:length(fileNames)
            handles.sparseMatFiles{fileInd}= matfile(fullfile(pathName,fileNames{fileInd}));
            handles.sparseMovies(fileInd,:)=handles.sparseMatFiles{fileInd}...
                .frames((1+frameOffsets(fileInd)):(frameOffsets(fileInd)+nFramesGood),1);
            md=handles.sparseMatFiles{fileInd}.metaData;
            handles.bgs(:,:,fileInd)=md.bg;
            % sparse images
%             fullIm=full(handles.sparseMovies(fileInd,1).image);
%             fullIm(fullIm==0)=intmax(class(handles.bgs));
            % indIm
            fullIm=handles.bgs(:,:,fileInd);
            frame=handles.sparseMatFiles{fileInd}.frames(1,1);
            fullIm(sub2ind(md.frameSize,frame.indIm(:,1),frame.indIm(:,2)))=frame.indIm(:,3);
            
            handles.hImage(fileInd)=imshow(fullIm,[intmin(class(fullIm)),intmax(class(fullIm))],'parent',handles.hAxes(fileInd));
        end
        [nFrames,~] = size(handles.sparseMatFiles{fileInd},'frames');
        
        handles.frameSlider.Min=1;
        handles.frameSlider.Max=nFrames;
        handles.frameSlider.Value=1;
        handles.frameSlider.SliderStep=[1,50]/(double(nFrames)-1);
        handles.textStartFrame.String=num2str(1);
        handles.textEndFrame.String=num2str(nFrames);
        handles.textCurrFrame.String=num2str(1);
        handles.frameSlider.Enable='on'; 
        handles.playToggle.Enable='on';
        handles.rewindToggle.Enable='on';
        handles.firstFrameButton.Enable='on';
        handles.lastFrameButton.Enable='on';
        handles.bgChkbx.Enable='on';
        handles.textCurrFrame.Enable='on';
        handles.textEditSkip.Enable='on';
        if ~isfield(handles.metaData,'frameRate')
            handles.metaData.frameRate=20000;
            warndlg('No frame rate saved in metaData, setting to 20000FPS!')
        end
        handles.textCurrMs.String=num2str(handles.metaData.startFrame*1000/handles.metaData.frameRate);
    end

    function ContinuousSlide(~, ~)
        handles.textCurrFrame.String=num2str(round(handles.frameSlider.Value));
        for fileInd=1:length(handles.sparseMatFiles)
            if handles.bgChkbx.Value
                newIm=handles.bgs(:,:,fileInd);
                newIm(sub2ind(handles.metaData.frameSize,...
                    handles.sparseMovies(fileInd,round(handles.frameSlider.Value)).indIm(:,1),...
                    handles.sparseMovies(fileInd,round(handles.frameSlider.Value)).indIm(:,2)))=...
                    handles.sparseMovies(fileInd,round(handles.frameSlider.Value)).indIm(:,3);
            else
                newIm=zeros(handles.metaData.frameSize,'like',handles.bgs);
                newIm(sub2ind(handles.metaData.frameSize,...
                    handles.sparseMovies(fileInd,round(handles.frameSlider.Value)).indIm(:,1),...
                    handles.sparseMovies(fileInd,round(handles.frameSlider.Value)).indIm(:,2)))=...
                    handles.sparseMovies(fileInd,round(handles.frameSlider.Value)).indIm(:,3);
            end
            set(handles.hImage(fileInd), 'CData', newIm);
        end
        handles.textCurrMs.String=['time: ',num2str((round(handles.frameSlider.Value)-1+handles.metaData.startFrame)...
            *1000/handles.metaData.frameRate),'ms'];
        drawnow limitrate
    end

    function CurrFrameEdit(~,~)
        frameNum = str2double(handles.textCurrFrame.String);
        if frameNum>=handles.frameSlider.Min&&frameNum<=handles.frameSlider.Max
            handles.frameSlider.Value= frameNum;
        else
           handles.textCurrFrame.String=num2str(handles.frameSlider.Value);
        end
    end
    
    function PlayPauseCallback(src,~)
        src.String='Pause';
        handles.rewindToggle.Enable='off';
        while src.Value
            frame2be= handles.frameSlider.Value+str2double(handles.textEditSkip.String);
            if frame2be>handles.frameSlider.Max
                handles.frameSlider.Value=handles.frameSlider.Min;
            else
                handles.frameSlider.Value=frame2be;
            end
        end
        handles.rewindToggle.Enable='on';
        src.String='Play';
    end
    
    function RewindPauseCallback(src,~)
        src.String='Pause';
        handles.playToggle.Enable='off';
        while src.Value
            frame2be= handles.frameSlider.Value-str2double(handles.textEditSkip.String);
            if frame2be<handles.frameSlider.Min
                handles.frameSlider.Value=handles.frameSlider.Max;
            else
                handles.frameSlider.Value=frame2be;
            end
        end
        handles.playToggle.Enable='on';
        src.String='Rewind';
    end
    
    function FirstFrameCallback(~,~)
        handles.frameSlider.Value=handles.frameSlider.Min;
    end

    function LastFrameCallback(~,~)
        handles.frameSlider.Value=handles.frameSlider.Max;
    end

    function SkipEdit(src,~)
        if isnan(str2double(src.String))
            src.String='1';
        end
    end

    function FigCloseReq(src,~)
        selection = questdlg('Close This Figure?',...
            'Close Request Function',...
            'Yes','No','Yes'); 
        switch selection
            case 'Yes'
                delete(src)
            case 'No'
                return 
        end
    end
end