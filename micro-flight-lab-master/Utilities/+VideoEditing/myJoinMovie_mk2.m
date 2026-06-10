function myJoinMovie_mk2(varargin)
    % generates joined movies of sorted triplets of cine/sparse files
    % optional inputs - myJoinMovie_mk2(frate_outp,quality_outp,alarm_flag)
    
    defaults = {30,25,true};
    defaults(1:nargin) = varargin;
    frate_outp=defaults{1};
    quality_outp=defaults{2};
    alarm_flag=defaults{3}; %make sound at end of execution

    [FileName,PathName] = uigetfile({'*.cine';'*.mat'},'Select the sparse file/s',...
                'MultiSelect', 'on','C:\Users\noamler\Downloads\all_cut_2018_05_21');
    if isequal(FileName,0)
        disp('User selected Cancel')
        return
    end
    if mod(length(FileName),3)~=0
        error('only multiples of three')
    else
        num_of_outp=length(FileName)/3;
        FileName=sort(FileName);
        isCin = strcmp(FileName{1}(end-3:end),'.cin') || strcmp(FileName{1}(end-4:end),'.cine');
        if isCin
            metaData=cell(length(FileName),1);
            cindata=cell(length(FileName),1);
            for i=1:length(FileName)
                metaData{i} = getCinMetaData(fullfile(PathName,FileName{i})) ;
                cindata{i}  = myOpenCinFile(fullfile(PathName,FileName{i})) ;
                realFrameRate=metaData{1}.framerate;
            end
        else
            movies=cell(length(FileName),1);
            for i=1:length(FileName)
                loady=load(fullfile(PathName,FileName{i})); %loads sparse_array
                movies{i}=loady.sparse_array;
                realFrameRate=20000; %todo: get from xml
            end
        end
    end

    for k=1:num_of_outp
        barename=strsplit(FileName{1+3*(k-1)},'.');
        movname=strsplit(barename{1},'_');
        outputVideo = VideoWriter([PathName,movname{1},...
            'join_q',num2str(quality_outp),'_fr',num2str(frate_outp)],'MPEG-4');%#ok
        outputVideo.FrameRate = frate_outp;
        outputVideo.Quality = quality_outp;
        open(outputVideo);
        skip_fr=1;
        frame_counter = -skip_fr ;
        if isCin
            first_frame=max(cellfun(@(x) extractfield(x,'firstImage'),metaData((1:3)+3*(k-1))));
            last_frame=max(cellfun(@(x) extractfield(x,'lastImage'),metaData((1:3)+3*(k-1))));
        else %sparse
            first_frame=1;
            last_frame=length(movies{1})-1;
        end
        Nframes = last_frame - first_frame + 1 ;

        for fr=first_frame:last_frame
            frame_counter = frame_counter + skip_fr ;
            if (mod(frame_counter,100)==0) % progression display
                disp([ num2str(frame_counter) ' / ' num2str(Nframes)]) ; 
            end
            if isCin
                im_cell=cellfun(@(x) myReadCinImage(x,fr),cindata((1:3)+3*(k-1)),'UniformOutput',0);
                [sw,se,n]=im_cell{:};
            else
                full_im=full(movies{1}{fr+1});
                new_im=double(movies{1}{1}); %background
                sw=new_im.*double(full_im==0)+full_im;
                full_im=full(movies{2}{fr+1});
                new_im=double(movies{2}{1}); %background
                se=new_im.*double(full_im==0)+full_im;
                full_im=full(movies{3}{fr+1});
                new_im=double(movies{3}{1}); %background
                n=new_im.*double(full_im==0)+full_im;
            end
            se =imresize(se,[size(sw,1),size(sw,2)]);
            n_temp = imresize(n,[size(sw,1),size(sw,2)]);
            blackImage = zeros(size(sw,1),size(sw,2)/2,size(sw,3));
            n=[blackImage,n_temp,blackImage];
            videoframe = uint8(double([n;sw,se])/256);
            realtimeMS = round(double(frame_counter) / realFrameRate * 1000) ;
            writeVideo(outputVideo,insertText(videoframe,[0,0],[num2str(realtimeMS),'ms'],...
                            'fontsize',60,'BoxColor','white','BoxOpacity',0.5,'TextColor','black'));
        end
        close(outputVideo);
    end

    cellfun(@(x) myCloseCinFile(x),cindata)

    if alarm_flag
        load chirp
        sound(y,Fs)
    end
end