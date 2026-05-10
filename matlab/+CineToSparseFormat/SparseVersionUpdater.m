function SparseVersionUpdater()
%{
Description:
-----------
- select files using gui
- saves output files in current directory (to avoid overwriting)
- updates old sparse formats to newest format:
    - unpack sparseStruct father struct
    - changes frames cell array to struct array
    - collects all metadata under the metadata field
    - converts sparse matrices to indIm format [row,col,val]
    - adds the frameSize to metaData
    - for older versions where the bg is the first frame in the cell array, it
        removes it and saves in the metadata struct

Example:
-------
SparseVersionUpdater()
%}
%% choose files in gui
    [fileNames,folderPath] = uigetfile('*.mat','Select sparse files to convert',...
        'MultiSelect', 'on');
    if isequal(fileNames,0)
       disp('User selected Cancel; end of session')
       return
    end
    % single file to cell
    if ~iscell(fileNames)
        fileNames={fileNames};
    end
    fileInd=0;
%% loop on files 
    while fileInd<length(fileNames)
        fileInd=fileInd+1;
        load(fullfile(folderPath,fileNames{fileInd}));
        % unfolding of sparseStruct
        if exist('sparseStruct','var')
            if isfield(sparseStruct,'frames')
                frames = sparseStruct.frames;
            end
            if isfield(sparseStruct,'bg')
                metaData.bg = sparseStruct.bg;
            end
            if isfield(sparseStruct,'startFrame')
                metaData.startFrame = sparseStruct.startFrame;
            end
            if isfield(sparseStruct,'frameRate')
                metaData.frameRate= sparseStruct.frameRate;
            end
        end
        % converts cell array of structs to struct array
        if exist('frames','var')
            if iscell(frames)
                frames = cell2mat(frames);
            end
        % converts cell array of sparse to struct array
        elseif exist('sparse_array','var')
            metaData.bg=sparse_array{1};
            frames=cell2struct(sparse_array(2:end),'image',2);
        end
        % creating the metaData struct
        if ~exist('metaData','var')
            if exist('frameRate','var')
                metaData.frameRate=frameRate;
            end
            if exist('bg','var')
                metaData.bg=bg;
            end
            if exist('startFrame','var')
                metaData.startFrame=startFrame;
            end
        end
        % add frame size to metadata
        if ~isfield(metaData,'frameSize')
            metaData.frameSize=size(metaData.bg);
        end
        % convert sparse image to uint16 [row,col,v]
        if isfield(frames,'image')
            for frameInd=1:length(frames)
                [row,col,v] = find(frames(frameInd).image);
                frames(frameInd).indIm=uint16([row,col,v]);
            end
            frames=rmfield(frames,'image');
        end
        % save in -v7.3 to allow partial loading using matfile
        save(fileNames{fileInd},'frames','metaData','-v7.3')
    end
end