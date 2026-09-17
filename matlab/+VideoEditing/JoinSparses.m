function outFile = JoinSparses(fileNames,varargin)
%{
Description:
-----------
- Generates and saves the raw movie of one sparse movie: its camera views
  tiled into a single frame, with the camera names and the trigger-relative
  frame number and time on screen.
- Frames are synchronized across cameras by metaData.startFrame and cut to
  the frames all cameras share.
- Each frame is rebuilt raw: the stored pixels pasted onto the camera's
  background, as SparseMovieLoader_4cam shows them. Mats are shown as
  stored, so a mirror cam flipped in place by flip_sparse_cam_mat shows
  flipped.
- Layout: 4 cams [1 2; 3 4], 3 cams [1 3; 2 centred], 2 cams [1 2],
  1 cam alone. A camera with a smaller frame is centred on black.
- Saved as <movie>_raw_fr<outFrameRate>_skip<skip>.mp4. It is written under
  a temporary name and renamed only once complete, so an interrupted run
  never leaves a truncated movie under the final name.
- Writes MPEG-4 directly where VideoWriter supports it (Windows, macOS).
  On Linux, where it does not, writes a Motion JPEG AVI to tempdir and
  converts it to an H.264 mp4 with ffmpeg.
- Merges Join3Sparses and Join4Sparses from the lab's MatlabUtils.

Input:
-----
fileNames - cell array with 1-4 sparse movie paths, or a movie folder
    holding 1-4 *_sparse.mat files.
outFrameRate (Optional) - frame rate of output movie. default 30.
outQuality (Optional) - VideoWriter quality, 0-100. default 90.
skip (optional name-value pair) - frame skipping. default 1.
outDir (optional name-value pair) - output folder. default: the folder of
    the sparse files.

Output:
------
outFile - path of the saved movie.

Example:
-------
VideoEditing.JoinSparses('/path/to/mov6',30,90,'skip',10)
%}
%% parse inputs and initialize variables
    inpParser = inputParser;
    addRequired(inpParser,'fileNames');
    addOptional(inpParser,'outFrameRate',30);
    addOptional(inpParser,'outQuality',90);
    addParameter(inpParser,'skip',1);
    addParameter(inpParser,'outDir','');
    parse(inpParser,fileNames,varargin{:});
    opts = inpParser.Results;

    if ischar(fileNames) || isstring(fileNames)
        listing = dir(fullfile(char(fileNames),'*_sparse.mat'));
        fileNames = fullfile({listing.folder},{listing.name});
    end
    nCams = numel(fileNames);
    if nCams < 1 || nCams > 4
        error('JoinSparses:nCams','expected 1-4 sparse movies, got %d',nCams);
    end

    camNames = cell(1,nCams);
    for camInd = nCams:-1:1
        loaded = load(fileNames{camInd},'frames','metaData');
        frames{camInd} = loaded.frames;
        metaDatas{camInd} = loaded.metaData;
        [~,name] = fileparts(fileNames{camInd});
        camNames{camInd} = regexp(name,'cam\d+','match','once','ignorecase');
        if isempty(camNames{camInd})
            camNames{camInd} = sprintf('file %d',camInd);
        end
    end

    % synchronize all movies and cut excess data
    startFrames = cellfun(@(md) md.startFrame,metaDatas);
    frameOffsets = max(startFrames)-startFrames;
    nFrames = min(cellfun(@numel,frames)-frameOffsets);
    if nFrames < 1
        error('JoinSparses:noOverlap','the sparse movies share no frames');
    end
    firstFrameNum = max(startFrames); % trigger-relative number of shared frame 1
    realFrameRate = metaDatas{1}.frameRate;

    % tile layout: top-left pixel (row,col) of each camera's tile
    frameSizes = cell2mat(cellfun(@(md) md.frameSize(:)',metaDatas,'UniformOutput',false)');
    H = max(frameSizes(:,1));
    W = max(frameSizes(:,2));
    switch nCams
        case 1
            tileOrigins = [0,0];
            canvasSize = [H,W];
        case 2
            tileOrigins = [0,0; 0,W];
            canvasSize = [H,2*W];
        case 3
            tileOrigins = [0,0; H,floor(W/2); 0,W];
            canvasSize = [2*H,2*W];
        case 4
            tileOrigins = [0,0; 0,W; H,0; H,W];
            canvasSize = [2*H,2*W];
    end
    if nCams == 3
        clockPosition = [floor(W/4),H+floor(H/2)]; % the empty bottom-left
        clockAnchor = 'Center';
    else
        clockPosition = [floor(canvasSize(2)/2),canvasSize(1)]; % bottom centre, off the camera tags
        clockAnchor = 'CenterBottom';
    end
    tagPositions = tileOrigins(:,[2,1]); % insertText takes [x,y]

    if isempty(opts.outDir)
        opts.outDir = fileparts(fileNames{1});
    end
    [~,name] = fileparts(fileNames{1});
    movName = regexprep(name,'_cam\d+.*$','','ignorecase');
    outFile = fullfile(opts.outDir,sprintf('%s_raw_fr%d_skip%d.mp4',...
        movName,opts.outFrameRate,opts.skip));
    useMpeg4 = any(strcmp({VideoWriter.getProfiles().Name},'MPEG-4'));
    if useMpeg4
        videoFile = fullfile(opts.outDir,['.',movName,'_raw_partial.mp4']);
        outputVideo = VideoWriter(videoFile,'MPEG-4');
    else
        videoFile = [tempname,'.avi'];
        outputVideo = VideoWriter(videoFile,'Motion JPEG AVI');
    end
    outputVideo.FrameRate = opts.outFrameRate;
    outputVideo.Quality = opts.outQuality;
    open(outputVideo);
%% loop on frames
    frameInds = 1:opts.skip:nFrames;
    progressEvery = max(1,round(numel(frameInds)/10));
    fprintf('%s: %d cams, %d frames, writing %d\n',movName,nCams,nFrames,numel(frameInds));
    for k = 1:numel(frameInds)
        frameInd = frameInds(k);
        % progression display
        if mod(k,progressEvery) == 0
            fprintf('  frame %u/%u\n',frameInd,nFrames);
        end
        canvas = zeros(canvasSize,'like',metaDatas{1}.bg);
        for camInd = 1:nCams
            md = metaDatas{camInd};
            im = md.bg;
            frame = frames{camInd}(frameInd+frameOffsets(camInd));
            im(sub2ind(md.frameSize,double(frame.indIm(:,1)),double(frame.indIm(:,2)))) = frame.indIm(:,3);
            % centre a smaller frame in its tile
            r0 = tileOrigins(camInd,1)+floor((H-md.frameSize(1))/2);
            c0 = tileOrigins(camInd,2)+floor((W-md.frameSize(2))/2);
            canvas(r0+(1:md.frameSize(1)),c0+(1:md.frameSize(2))) = im;
        end
        videoframe = im2uint8(canvas); % writeVideo doesn't work with uint16
        frameNum = firstFrameNum+frameInd-1;
        % add time and camera tags
        writeVideo(outputVideo,insertText(insertText(...
            videoframe,clockPosition,sprintf('frame %d   %.2f ms',frameNum,frameNum*1000/realFrameRate),...
            'FontSize',40,'BoxColor','white','BoxOpacity',1,'TextColor','black','AnchorPoint',clockAnchor),...
            tagPositions,camNames,...
            'FontSize',30,'BoxColor','white','BoxOpacity',0.2,'TextColor','black'));
    end
    close(outputVideo);

    if useMpeg4
        movefile(videoFile,outFile);
    else
        % MATLAB's libraries on LD_LIBRARY_PATH can break the system ffmpeg
        partFile = [outFile,'.part'];
        [status,msg] = system(sprintf(['env LD_LIBRARY_PATH= ffmpeg -y -loglevel error -i "%s" ',...
            '-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart -f mp4 "%s"'],...
            videoFile,partFile));
        delete(videoFile); % the AVI is GBs; never leave it behind
        if status ~= 0
            if isfile(partFile)
                delete(partFile);
            end
            error('JoinSparses:ffmpeg','ffmpeg failed to write %s:\n%s',outFile,msg);
        end
        movefile(partFile,outFile);
    end
    fprintf('saved %s\n',outFile);
end
