clearvars -except num_cams max_frames sparse_folder_path save_path movie_num start_ind end_ind

%% set paths
% load('C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\best_frames_21-7.mat') % load dataset (mov|frame) list
% sparse_folder_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder'; % folder with sparse movies

% sparse_folder_path = "G:\My Drive\Amitai\experiment magnet + UV 30.8\movies\";
if ~exist('sparse_folder_path','var') || isempty(sparse_folder_path)
    sparse_folder_path = "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/inference_datasets/new_roni_experiments/2022_02_03/hull/hull_Reorder_test";
end
if ~exist('save_path','var') || isempty(save_path)
    save_path = sparse_folder_path;
end

%% set a apecific movie
if ~exist('movie_num','var') || isempty(movie_num)
    movie_num = 1;
end

if ~exist('start_ind','var') || isempty(start_ind)
    start_ind = 8;
end
if ~exist('end_ind','var') || isempty(end_ind)
    % auto-detect from the first sparse mat's frames count
    auto_files = dir(fullfile(sparse_folder_path, ['mov', int2str(movie_num)], '*sparse.mat'));
    if isempty(auto_files)
        error('No *sparse.mat files found in %s', fullfile(sparse_folder_path, ['mov', int2str(movie_num)]));
    end
    auto_mat = matfile(fullfile(auto_files(1).folder, auto_files(1).name));
    end_ind = size(auto_mat, 'frames', 1) - 7;   % subtract time_jump margin
    fprintf('end_ind auto-detected as %d (from %s)\n', end_ind, auto_files(1).name);
end

% Optional: limit to the first max_frames frames (useful for quick test runs).
% Set here, or pass via CLI: matlab -batch "max_frames=400; run('...')"
% Leave unset (or empty) to process the full [start_ind, end_ind] range.
if ~exist('max_frames','var') || isempty(max_frames)
    max_frames = [];   % default: no limit
end
if ~isempty(max_frames)
    end_ind = min(end_ind, start_ind + max_frames - 1);
    fprintf('max_frames=%d -> end_ind clamped to %d\n', max_frames, end_ind);
end

best_frames_mov_idx = zeros(end_ind - start_ind + 1, 2);
best_frames_mov_idx(:, 2) = (start_ind:end_ind);
best_frames_mov_idx(:, 1) = movie_num;
num_frames=size(best_frames_mov_idx,1);

%%
num_masks = 0;
if ~exist('num_cams','var') || isempty(num_cams)
    num_cams = 4;   % default; override from CLI: matlab -r "num_cams=3; run('...'); exit"
end
crop_size=192*[1,1];


%% change time channels
time_jump=7;
num_time_channels=3;
frame_time_offsets=linspace(-time_jump,time_jump,num_time_channels);

num_channels=num_cams*(num_time_channels + num_masks);
data=zeros([crop_size,num_channels],'single');
tic

save_name=fullfile(save_path,['mov_', ...
    num2str(movie_num),'_', ...
    num2str(start_ind),'_', ...
    num2str(end_ind),'_','ds_',...
    num2str(num_time_channels),'tc_',...
    num2str(time_jump),'tj.h5']);

% create 3 datasets:
% - box - holds the cropped images for all cameras
% - cropzone - holds the top left coordinate of the cropped window
% - frameInds - holds the frame indices for synchronization/testing
%% save best_frames_mov_idx
h5create(save_name,'/best_frames_mov_idx',size(best_frames_mov_idx))
h5write(save_name,'/best_frames_mov_idx',best_frames_mov_idx);

%% create the other datasets
h5create(save_name,'/box',[crop_size,num_channels,Inf],'ChunkSize',[crop_size,num_channels,1],...
    'Datatype','single','Deflate',1)
h5create(save_name,'/cropzone',[2,num_cams,Inf],'ChunkSize',[2,num_cams,1],...
    'Datatype','uint16','Deflate',1)
h5create(save_name,'/frameInds',[1,num_cams,Inf],'ChunkSize',[1,num_cams,1],...
    'Datatype','uint16','Deflate',1)

%% loop on frames
% Detect whether we're running under SLURM (sbatch). In that case the output
% is a log file, not a TTY, and backspace characters don't erase — they just
% accumulate into one giant line. Switch to a quieter progress format: print
% one newline-terminated line every PROGRESS_EVERY frames.
batch_progress = ~isempty(getenv('SLURM_JOB_ID'));
PROGRESS_EVERY = 250;
fprintf('\n');
if batch_progress
    line_length = 0;
    fprintf('frame: 0/%u\n', num_frames);
else
    line_length = fprintf('frame: %u/%u',0,num_frames);
end
h5_ind=0;
% load frames

% mov_num=sprintf('%d',best_frames_mov_idx(frame_ind,1));
% start_frame=best_frames_mov_idx(frame_ind,2);
full_file_name = fullfile(sparse_folder_path,['mov',int2str(movie_num)]);
file_names =  dir(full_file_name);
file_names = {file_names.name};
file_names_sparse = [];
for name=1:size(file_names,2)
%         disp(file_names(name))
    if endsWith(file_names(name), 'sparse.mat')
        file_names_sparse = [file_names_sparse, file_names(name)];
    end
end
file_names = file_names_sparse;
mf = cellfun(@(x) matfile(fullfile(sparse_folder_path,['mov',int2str(movie_num)],x)),file_names,...
        'UniformOutput',false);
all_meta_data= cellfun(@(x) x.metaData,mf);
frames=cellfun(@(x) x.frames((start_ind-time_jump):(end_ind+time_jump),1),mf,'UniformOutput',false);


for frame_ind=(1+time_jump):(num_frames+time_jump)
    if batch_progress
        % print every PROGRESS_EVERY-th frame and on the last frame
        actual = frame_ind - time_jump;
        if mod(actual, PROGRESS_EVERY) == 0 || frame_ind == (num_frames + time_jump)
            fprintf('frame: %u/%u\n', actual, num_frames);
        end
    else
        fprintf(repmat('\b',1,line_length))
        line_length = fprintf('frame: %u/%u',frame_ind,num_frames);
    end


    %% loop on cameras
    for cam_ind=num_cams:-1:1
        frame=frames{cam_ind}(frame_ind);
        % keep only largest blob
        full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
        lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
        full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3); % using the "negative" of the mosquito
        full_im(~bwareafilt(full_im>0,1))=0;
        [r,c,v] = find(full_im);
        frame.indIm=[r,c,v];
        % skip camera if no insect blob detected in this frame
        if isempty(r)
            data(:,:, num_time_channels*(cam_ind-1)+1 : num_time_channels*cam_ind) = 0;
            crop_zone_data(:,cam_ind) = uint16([1; 1]);
            continue
        end
        % blob boundaries
        max_find_row=double(max(frame.indIm(:,1)));
        min_find_row=double(min(frame.indIm(:,1)));
        max_find_col=double(max(frame.indIm(:,2)));
        min_find_col=double(min(frame.indIm(:,2)));

        % pad blob bounding box to reach crop_size
        row_pad=crop_size(1)-(max_find_row-min_find_row+1);
        col_pad=crop_size(2)-(max_find_col-min_find_col+1);
        if (floor(min_find_row-row_pad/2) < 1)
            row_offset = 1-floor(min_find_row-row_pad/2);
        elseif (floor(max_find_row+row_pad/2)> all_meta_data(cam_ind).frameSize(1))
            row_offset = all_meta_data(cam_ind).frameSize(1)-floor(max_find_row+row_pad/2);
        else
            row_offset = 0;
        end
        if (floor(min_find_col-col_pad/2) < 1)
            col_offset = 1-floor(min_find_col-col_pad/2);
        elseif (floor(max_find_col+col_pad/2)> all_meta_data(cam_ind).frameSize(2))
            col_offset = all_meta_data(cam_ind).frameSize(2)-floor(max_find_col+col_pad/2);
        else
            col_offset = 0;
        end
        %% loop on extra time frames (future and past)
        offset_counter=length(frame_time_offsets);
        for frameOffset=frame_time_offsets
            frame=frames{cam_ind}(frame_ind+frameOffset);
%             frames_offs=cellfun(@(x) x.frames(start_frame+frameOffset,1),mf,'UniformOutput',false);
%             frame=frames_offs{cam_ind};
            full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
            lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
            full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3);
            % normalize (consistent with trainng data for NN) after cropping
            data(:,:,num_time_channels*cam_ind-offset_counter+1)=mat2gray(full_im((floor(min_find_row-row_pad/2):floor(max_find_row+row_pad/2))+row_offset...
                ,(floor(min_find_col-col_pad/2):floor(max_find_col+col_pad/2))+col_offset));
            offset_counter= offset_counter-1;
        end
        crop_zone_data(:,cam_ind)=uint16([floor(min_find_row-row_pad/2)+row_offset;...
                floor(min_find_col-col_pad/2)+col_offset]);
    end
    h5_ind=h5_ind+1;
    h5write(save_name,'/box',im2single(data),[1,1,1,h5_ind],[crop_size,num_channels,1]);
    h5write(save_name,'/cropzone',crop_zone_data,[1,1,h5_ind],[2,num_cams,1]);
    h5write(save_name,'/frameInds',uint16(frame_ind*ones(1,num_cams)),[1,1,h5_ind],[1,num_cams,1]);
end
fprintf('\n')
disp([save_name,' dataset was created. ',num2str(toc),' Sec'])