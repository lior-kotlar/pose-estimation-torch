function myMakeMovie_batch_ui()
% converts multiple cine files to mp4 files with running timer
    format compact

    % add cine funtions to path
    run('C:\google drive\Micro Flight Group\Code\MatlabCode\PhantomSDK\runMeFirst.m')

    % select parameters
    realFrameRate = 10000 ;
    skip_fr=1;
    frate_outp=30;  % Default 30
    quality_outp=25;    % Default 75

    % ui load files
    [mov_fnames,mov_path,~] = uigetfile('*.*','Select Movie','MultiSelect','on');
    if ~iscell(mov_fnames)   % avoids one-file errors
        mov_fnames = {mov_fnames};
    end

    % loop on all vids
    for i=1:length(mov_fnames)
        filename = mov_fnames{i};
        % create output video
        barename=strsplit(filename,'.');
        out = VideoWriter([mov_path,barename{1},'_q',num2str(quality_outp)],'MPEG-4') ;
        out.FrameRate = frate_outp;
        out.Quality   = quality_outp;
        open(out) ;
        frame_counter = -skip_fr ;
        disp(['file: ',filename]) ; 
        switch barename{2}
            case 'cine'
            cindata = myOpenCinFile([mov_path filename]);
            first_frame = cindata.firstIm ;
            last_frame = cindata.lastIm ;
            Nframes = last_frame - first_frame + 1 ;
            for k=first_frame:skip_fr:last_frame
                frame_counter = frame_counter + skip_fr ;
                img = myReadCinImage (cindata, k);
                % cast to uint8 if needed
                if (isa( img(1,1),'uint16'))
                    img = uint8(double(img)/256) ; 
                end
                realtimeMS = round(double(frame_counter) / realFrameRate * 1000) ;
                % write frame with timer
                writeVideo(out,insertText(img,[0,0],[num2str(realtimeMS),'ms'],...
                    'fontsize',12,'BoxColor','white','BoxOpacity',0.5,'TextColor','black'));
                if (mod(frame_counter,100)==0) % progression display
                    disp([ num2str(frame_counter) ' / ' num2str(Nframes)]) ; 
                end 
            end
            myCloseCinFile(cindata);
        end
        close(out);
    end
end