clear

%% set paths
% load('C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\best_frames_21-7.mat') % load dataset (mov|frame) list
% sparse_folder_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder'; % folder with sparse movies

sparse_folder_path_read = "C:\Users\amita\OneDrive\Desktop\temp\next_batch_movies\";
drive_save = 'C:\Users\amita\OneDrive\Desktop\temp\next_batch_movies';

for mov_num = [59, 60]
    mov_num
    save_path = fullfile(drive_save, ['mov', int2str(mov_num)]);
    
    % Check if save_path is a real directory
    if isfolder(save_path)
        % Extract start_frame and end_frame from README_mov{mov_num}.txt
        [start_frame, end_frame] = extract_frame_info(save_path, mov_num);
        % end_frame = end_frame
        create_H5(mov_num, start_frame, end_frame, sparse_folder_path_read, save_path, drive_save);
    else
        fprintf('Directory does not exist: %s\n', save_path);
    end
end


function [start_frame, end_frame] = extract_frame_info(save_path, mov_num)
    % Construct the full file path for the README file
    file_name = sprintf('README_mov%d.txt', mov_num);
    file_path = fullfile(save_path, file_name);
    
    % Open the file and read the contents
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Failed to open the file: %s', file_path);
    end
    
    % Initialize variables
    start_frame = [];
    end_frame = [];
    
    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'start:')
            start_frame = sscanf(line, 'start: %d');
        elseif contains(line, 'finish:')
            end_frame = sscanf(line, 'finish: %d');
        end
    end
    
    % Close the file
    fclose(fid);
    
    % Check if the values were successfully read
    if isempty(start_frame) || isempty(end_frame)
        error('Failed to extract start_frame and end_frame from the file: %s', file_path);
    end
end

function create_H5(movie_num, start_frame, end_frame, sparse_folder_path, save_path, drive_path)
%% set a apecific movie
    % movie_num = 1;
    stop=false;
    start_ind = start_frame;
    end_ind = end_frame;
    best_frames_mov_idx = zeros(end_ind - start_ind + 1, 2);
    best_frames_mov_idx(:, 2) = (start_ind:end_ind);
    best_frames_mov_idx(:, 1) = movie_num;
    num_frames=size(best_frames_mov_idx,1);
    
    %%
    num_masks = 0;
    num_cams=4;
    crop_size=192*[1,1];
    
    
    %% change time channels
    time_jump=7;
    num_time_channels=3;
    frame_time_offsets=linspace(-time_jump,time_jump,num_time_channels);
    
    num_channels=num_cams*(num_time_channels + num_masks);
    data=zeros([crop_size,num_channels],'single');
    tic
    
    save_name=fullfile(save_path,['movie_', ...
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
    fprintf('\n');
    line_length = fprintf('frame: %u/%u',0,num_frames);
    h5_ind=0;
    % load frames
    
    % mov_num=sprintf('%d',best_frames_mov_idx(frame_ind,1));
    % start_frame=best_frames_mov_idx(frame_ind,2);
    full_file_name = fullfile(sparse_folder_path,['mov',int2str(movie_num)]);
    % full_file_name
    file_names =  dir(full_file_name);
    file_names = {file_names.name};
    % file_names
    file_names_sparse = [];
    for name=1:size(file_names,2)
    %         disp(file_names(name))
        if endsWith(file_names(name), 'sparse.mat')
            file_names_sparse = [file_names_sparse, file_names(name)];
        end
    end
    file_names = file_names_sparse;
    % file_names
    movie_ = ['mov', int2str(movie_num)];
    mf = [];
    for i = 1:length(file_names)
        mf{i} = matfile(fullfile(sparse_folder_path, movie_, file_names{i}));
    end
    % all_meta_data = cellfun(@(x) x.metaData,mf);
    % mf
    all_meta_data = struct('bg', {}, 'startFrame', {}, 'frameRate', {}, 'frameSize', {}, 'isFlipped', {}, 'xmlStruct', {});
    for i = 1:length(mf)
        all_meta_data(i) = mf{i}.metaData;
    end

    % frames=cellfun(@(x) x.frames((start_ind-time_jump):(end_ind+time_jump),1),mf,'UniformOutput',false);
    
    size(mf);

    frames = cell(size(mf));
    for i = 1:length(mf)
        frames{i} = mf{i}.frames((start_ind - time_jump):(end_ind + time_jump), 1);
    end

    frames;

    last_fly_CM = zeros(num_cams, 2);
    last_fly_pixels_sum = zeros(num_cams,1);
    for frame_ind=(1+time_jump):(num_frames+time_jump)
        fprintf(repmat('\b',1,line_length))
        line_length = fprintf('frame: %u/%u',frame_ind,num_frames);
        
        %% loop on cameras
        for cam_ind=num_cams:-1:1
            

            frame=frames{cam_ind}(frame_ind);
            % keep only largest blob
            full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
            lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
            full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3); % using the "negative" of the mosquito
            
            % if the frame is not the first one, choose te connectivity
            % component that is closest to the previous frame ceenter of
            % mass as the 

            CC = bwconncomp(full_im);
            if CC.NumObjects > 1
                
                
                pixelSums = cellfun(@(x) sum(full_im(x)), CC.PixelIdxList);
                % numPixels = cellfun(@numel,CC.PixelIdxList);
                [B, I] = sort(pixelSums, 'descend');
                inds = I(1:2); 

                mask1 = zeros(size(full_im));
                mask1(CC.PixelIdxList{inds(1)}) = 1;
                [r, c] = find(mask1);
                CM1 = [mean(c), mean(r)];
    
                mask2 = zeros(size(full_im));
                mask2(CC.PixelIdxList{inds(2)}) = 1;
                [r, c] = find(mask2);
                CM2 = [mean(c), mean(r)];
    
                dist1 = norm(CM1 - last_fly_CM(cam_ind, :)); 
                dist2 = norm(CM2 - last_fly_CM(cam_ind, :)); 
                if dist1 < dist2
                    mask = mask1 > 0;
                else
                    mask = mask2 > 0;
                end

                full_im_1 = bsxfun(@times, full_im, cast(mask1, 'like', full_im));
                full_im_2 = bsxfun(@times, full_im, cast(mask2, 'like', full_im));
                
                % find sharpness of image using the gradients
                kx= [1 ,0 ,-1; 2,0,-2; 1, 0 ,-1];
                ky= [1,2,1; 0,0, 0; -1, -2 ,-1];
    
                H = conv2(im2double(full_im_1),kx,'same');
                V = conv2(im2double(full_im_1),ky,'same');
                E1 = sqrt(H.*H + V.*V);
                % imshow(E1, [])

                H = conv2(im2double(full_im_2),kx,'same');
                V = conv2(im2double(full_im_2),ky,'same');
                E2 = sqrt(H.*H + V.*V);
                % imshow(E2, [])

                sharpness_1 = sum(E1(:));
                sharpness_2 = sum(E2(:));

                % find size of blob
                sum1 = sum(full_im_1(:));
                sum2 = sum(full_im_2(:));

                % scores

                score1 = sum1 / 500000 + sharpness_1 - dist1;
                score2 = sum2 / 500000 + sharpness_2 - dist2;

                if score1 > score2
                    full_im = full_im_1;
                    last_fly_pixels_sum(cam_ind) = sum1;
                else
                    full_im = full_im_2;
                    last_fly_pixels_sum(cam_ind) = sum2;
                end

                % full_im = bsxfun(@times, full_im, cast(mask, 'like', full_im));
            else
                full_im(~bwareafilt(full_im>0,1))=0;
            end
            % center of mass of the fly
            [r, c] = find(full_im);
            CM = [mean(c), mean(r)];
            last_fly_CM(cam_ind, :) = CM;

            [r,c,v] = find(full_im);
            frame.indIm=[r,c,v];
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
                try
                    data(:,:,num_time_channels*cam_ind-offset_counter+1)=mat2gray(full_im((floor(min_find_row-row_pad/2):floor(max_find_row+row_pad/2))+row_offset...
                    ,(floor(min_find_col-col_pad/2):floor(max_find_col+col_pad/2))+col_offset));
                catch
                    stop=true;
                    break
                end
                offset_counter= offset_counter-1;
            end
            if stop
                break
            end
            crop_zone_data(:,cam_ind)=uint16([floor(min_find_row-row_pad/2)+row_offset;...
                    floor(min_find_col-col_pad/2)+col_offset]);
        end
        if stop
            break
        end
        if any(crop_zone_data(:) > 1009)
            a=1;
        end
        h5_ind=h5_ind+1;
        h5write(save_name,'/box',im2single(data),[1,1,1,h5_ind],[crop_size,num_channels,1]);
        h5write(save_name,'/cropzone',crop_zone_data,[1,1,h5_ind],[2,num_cams,1]);
        h5write(save_name,'/frameInds',uint16(frame_ind*ones(1,num_cams)),[1,1,h5_ind],[1,num_cams,1]);
    end
    if stop
        save_name_new=fullfile(save_path,['movie_', ...
        num2str(movie_num),'_', ...
        num2str(start_ind),'_', ...
        num2str(frame_ind),'_','ds_',...
        num2str(num_time_channels),'tc_',...
        num2str(time_jump),'tj.h5']);
        movefile(save_name,save_name_new);
        save_name = save_name_new;
    end
   
    new_save_drive = fullfile(drive_path,['\mov', int2str(movie_num),'\movie_', ...
        num2str(movie_num),'_', ...
        num2str(start_ind),'_', ...
        num2str(frame_ind),'_','ds_',...
        num2str(num_time_channels),'tc_',...
        num2str(time_jump),'tj.h5']);
    movefile(save_name, new_save_drive)
    fprintf('\n')
    disp([save_name,' dataset was created. ',num2str(toc),' Sec'])
end