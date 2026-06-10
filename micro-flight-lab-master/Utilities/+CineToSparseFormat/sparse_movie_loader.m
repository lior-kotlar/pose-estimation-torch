function sparse_movie_loader()
    global sparse_movies handles_struct cindata
    handles_struct=struct;
    createFigureAndAxes();
    insertButtons();
    
    function createFigureAndAxes()
        % Close figure opened by last run
        figTag = 'CVST_VideoOnAxis_9804532';
        close(findobj('tag',figTag));

        % Create new figure
        handles_struct.hFig = figure('numbertitle', 'off', ...
               'name', 'Video In Custom GUI', ...
               'menubar','none', ...
               'toolbar','figure', ...
               'resize', 'on', ...
               'tag',figTag, ...
               'renderer','painters', ...
               'units','normalized',...
               'outerposition',[0 0 1 1],...
               'HandleVisibility','callback',...
               'CloseRequestFcn',@my_closereq); % hide the handle to prevent unintended modifications of our custom UI
        
        sreen_res=get(0,'ScreenSize'); %get screen resolution
        w=0.33;
        h=w/1.6*sreen_res(3)/sreen_res(4);
        first_line_y=0.6;
        % Create axes and titles
        handles_struct.hAxes{1} = createPanelAxisTitle(handles_struct.hFig,[0,first_line_y,w,h],'20302'); % [X Y W H]
        handles_struct.hAxes{2} = createPanelAxisTitle(handles_struct.hFig,[w,first_line_y,w,h],'20303');
        handles_struct.hAxes{3} = createPanelAxisTitle(handles_struct.hFig,[2*w,first_line_y,w,h],'20304');
        handles_struct.hAxes{4} = createPanelAxisTitle(handles_struct.hFig,[0,first_line_y-1.1*h,w,h],'cin-20302'); % [X Y W H]
        handles_struct.hAxes{5} = createPanelAxisTitle(handles_struct.hFig,[w,first_line_y-1.1*h,w,h],'cin-20303');
        handles_struct.hAxes{6} = createPanelAxisTitle(handles_struct.hFig,[2*w,first_line_y-1.1*h,w,h],'cin-20304');        
%         zoom(handles_struct.hFig,'on')
%         pan(handles_struct.hFig,'on')
    end

    function hAxis = createPanelAxisTitle(hFig, pos, axisTitle)

        % Create panel
        hPanel = uipanel('parent',hFig,'Position',pos,'Units','Normalized');
        % Create axis
        hAxis = axes('position',[0 0 1 1],'Parent',hPanel);
        hAxis.XTick = [];
        hAxis.YTick = [];
        hAxis.XColor = [1 1 1];
        hAxis.YColor = [1 1 1];
        % Set video title using uicontrol. uicontrol is used so that text
        % can be positioned in the context of the figure, not the axis.
        titlePos = [pos(1) pos(2)+pos(4)+0.01 pos(3) 0.02];
        uicontrol('style','text',...
            'String', axisTitle,...
            'Units','Normalized',...
            'Parent',hFig,'Position', titlePos,...
            'BackgroundColor',hFig.Color);

    end

    function insertButtons()
        handles_struct.Text_start = uicontrol(handles_struct.hFig,'style','text',...
            'Units','Normalized',...
            'string','0',...
            'position',[0.18 0.1 0.04 0.02]);
        handles_struct.Text_end = uicontrol(handles_struct.hFig,'style','text',...
            'Units','Normalized',...
            'string','0',...
            'position',[0.8 0.1 0.04 0.02]);
        handles_struct.Textframe = uicontrol(handles_struct.hFig,'style','Edit',...
            'Units','Normalized',...
            'string','0',...
            'position',[0.4 0.13 0.05 0.02],...
            'Enable','off',...
            'callback',@edit_frame_callback);
        handles_struct.sld = uicontrol(handles_struct.hFig,'Style', 'slider',...
            'Units','Normalized',...
            'Min',0,'Max',0,'Value',0,...
            'Position', [0.25 0.1 0.5 0.02],'Enable','off');
        addlistener(handles_struct.sld, 'Value', 'PostSet',@cont_sld);
        
        uicontrol(handles_struct.hFig,'style','pushbutton','string','LoadCmp',...
        'Units','Normalized','position',[0.1,0.05,0.05,0.04],'callback', ...
        @LoadCmpCallback);
    
        uicontrol(handles_struct.hFig,'style','pushbutton','string','Load',...
        'Units','Normalized','position',[0.1,0.1,0.05,0.04],'callback', ...
        @LoadCallback);
        handles_struct.play_btn=uicontrol(handles_struct.hFig,'style','togglebutton','string','Play',...
        'Units','Normalized','position',[0.02,0.1,0.05,0.04],...
        'Enable','off','Value',1,'callback', ...
        @PlayPauseCallback);
        handles_struct.bg_chkbx=uicontrol(handles_struct.hFig,'style','checkbox','string','Add bg',...
        'Units','Normalized','position',[0.02,0.15,0.05,0.02],...
        'Enable','off','Value',0,'callback', ...
        @BGCheckboxCallback);
        handles_struct.bg_flag=0;
        
        handles_struct.Text_mov_name = uicontrol(handles_struct.hFig,'style','text',...
            'Units','Normalized',...
            'string','Movie Name:',...
            'position',[0.72 0.15 0.25 0.02]);
    end

    function BGCheckboxCallback(src,~)
        if src.Value==1
            handles_struct.bg_flag=1;
        else
            handles_struct.bg_flag=0;
        end
        cont_sld(0,0);
    end

    function LoadCallback(~,~)
        [FileName,PathName] = uigetfile('*.mat','Select the sparse file/s',...
            'MultiSelect', 'on','C:\Users\noamler\Downloads\all_cut_2018_05_21');
        if isequal(FileName,0)
           disp('User selected Cancel')
           return
        end
        if ~iscell(FileName)
            FileName={FileName};
        end
        if length(FileName)>3
            %%add - error
            return
        end
        handles_struct.cmp_flag=0;
        name_split=strsplit(FileName{1},'_');
        handles_struct.Text_mov_name.String=['Movie Name: ',name_split{1}];
        sparse_movies=cell(length(FileName),1);
        for i=1:length(FileName)
            loady=load(fullfile(PathName,FileName{i})); %loads sparse_array
            loady_cell = struct2cell(loady);
%             sparse_movies{i} = loady_cell{1};
            sparse_movies{i}=loady.sparse_array;
            full_im=full(sparse_movies{i}{2});
            full_im(full_im==0)=1;
            handles_struct.himage{i}=imshow(full_im,[0,2^16-1],'parent',handles_struct.hAxes{i});
        end
        handles_struct.sld.Min=1;
        handles_struct.sld.Max=length(sparse_movies{i})-1;
        handles_struct.sld.Value=1;
        handles_struct.sld.Enable='on';
        Nframes=length(sparse_movies{i})-1;
        handles_struct.sld.SliderStep=[1,50]/(double(Nframes)-1);
        handles_struct.Text_start.String=num2str(1);
        handles_struct.Text_end.String=num2str(length(sparse_movies{i})-1);
        handles_struct.Textframe.String=num2str(1);
        handles_struct.play_btn.Enable='on';
        handles_struct.bg_chkbx.Enable='on';
        handles_struct.Textframe.Enable='on';
    end

    function LoadCmpCallback(~,~)
        folder_name =uigetdir();
        if isequal(FileName,0)
           disp('User selected Cancel')
           return
        end
        answer = inputdlg('Enter movie name:');
        %add -  check answer is legal
        mov_name=answer{1};
        Filenames=extractfield(dir(folder_name),'name');
        mov_filenames=Filenames(cell2mat(cellfun(@(x) contains(x,mov_name),...
            Filenames,'UniformOutput',0)));
        cine_filenames=mov_filenames(cell2mat(cellfun(@(x) contains(x,'.cine'),...
            mov_filenames,'UniformOutput',0)));
        sparse_filenames=mov_filenames(cell2mat(cellfun(@(x) contains(x,'.mat'),...
            mov_filenames,'UniformOutput',0)));
        if numel(cine_filenames)~=numel(sparse_filenames)
            %%error
            return
        end
        handles_struct.cmp_flag=1;
        metaData = getCinMetaData(fullfile(folder_name,cine_filenames{1})) ;
        for i=1:numel(sparse_filenames)
            loady=load(fullfile(folder_name,sparse_filenames{i})); %loads sparse_array
            sparse_movies{i}=loady.sparse_array;
            full_im=full(sparse_movies{i}{2});
            full_im(full_im==0)=2^16-1;
            handles_struct.himage{i}=imshow(full_im,[0,2^16-1],'parent',handles_struct.hAxes{i});
            
            cindata{i}  = myOpenCinFile(fullfile(folder_name,cine_filenames{i})) ;
            cin_im=myReadCinImage(cindata{i}, metaData.firstImage);
            handles_struct.himage{i+3}=imshow(cin_im,[0,2^16-1],'parent',handles_struct.hAxes{i+3});
        end
        handles_struct.sld.Min=metaData.firstImage;
        handles_struct.sld.Max=metaData.lastImage;
        handles_struct.sld.Value=metaData.firstImage;
        handles_struct.sld.Enable='on';
        Nframes=metaData.lastImage-metaData.firstImage+1;
        handles_struct.sld.SliderStep=[1,50]/(double(Nframes)-1);
        handles_struct.Text_start.String=num2str(metaData.firstImage);
        handles_struct.Text_end.String=num2str(metaData.lastImage);
        handles_struct.Textframe.String=num2str(metaData.firstImage);
        handles_struct.Text_mov_name.String=['Movie Name: ',mov_name];
        handles_struct.play_btn.Enable='on';
        handles_struct.bg_chkbx.Enable='on';
        handles_struct.Textframe.Enable='on';
    end

    function PlayPauseCallback(src,~)
        if src.Value==0
            handles_struct.play_flag=1;
            handles_struct.sld.Enable='off';
            handles_struct.Textframe.Enable='off';
            src.String='Pause';
            while handles_struct.play_flag==1
                if handles_struct.sld.Value==handles_struct.sld.Max
                    handles_struct.sld.Value=handles_struct.sld.Min;
                else
                    handles_struct.sld.Value= handles_struct.sld.Value+1;
                end
                cont_sld(0,0);
            end
            src.String='Play';
            handles_struct.sld.Enable='on';
            handles_struct.Textframe.Enable='on';
        else
            handles_struct.play_flag=0;
        end
    end
    
    function cont_sld(~, ~)
        frame_number   = floor(handles_struct.sld.Value);
        handles_struct.sld.Value=frame_number;
        set(handles_struct.Textframe,'String', num2str(frame_number));
        sparse_movie_frame=frame_number-handles_struct.sld.Min+1;
        for i=1:length(sparse_movies)
            full_im=full(sparse_movies{i}{sparse_movie_frame+1});
            if handles_struct.bg_flag
                new_im=double(sparse_movies{i}{1}); %background
                new_im=new_im.*double(full_im==0)+full_im;
                set(handles_struct.himage{i}, 'CData', new_im);
            else
                full_im(full_im==0)=2^16-1;
                set(handles_struct.himage{i}, 'CData', full_im);
            end
            if handles_struct.cmp_flag==1
                cin_im=myReadCinImage(cindata{i}, frame_number);
                set(handles_struct.himage{i+3}, 'CData', cin_im);
            end
            drawnow limitrate
        end
    end

    function edit_frame_callback(~,~)
        frame_number = str2double(handles_struct.Textframe.String);
        if frame_number>=handles_struct.sld.Min&&frame_number<=handles_struct.sld.Max
            handles_struct.sld.Value= frame_number;
            cont_sld(0,0);
        else
            handles_struct.Textframe.String=num2str(handles_struct.sld.Value);
        end
    end

    function my_closereq(src,~)
        selection = questdlg('Close This Figure?',...
            'Close Request Function',...
            'Yes','No','Yes'); 
        switch selection
            case 'Yes'
                if ~isempty(cindata)
                    cellfun(@(x) myCloseCinFile(x),cindata)
                end
                delete(src)
            case 'No'
                return 
        end
    end
end