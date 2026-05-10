clear
close all
clc
tic
% user input:
folder_path='D:\Omri\project\magnetic\cine\New folder'; % path of video files
cam_str='cam'; % letters before number of camerc (usually cam of cam_)
background_file_name=[]; % use for saved bg


% get list of names of files and initialize------------------------
files_name = dir([folder_path,'\*.cine']);
array_names = struct2cell(files_name);
number  = cell2mat(regexpi(array_names(1,:), 'cam','end'));  %the regexp extract the first string of digits from each char array.
cam_num = str2num(cell2mat(cellfun(@(x,y) x(y+1),array_names(1,:),num2cell(number,[1,length(number)]),'UniformOutput',0)'));
[sort_cam_num, ind]=sort(cam_num); % sort indices of camera number
[val_uni, ind_uni]=unique(sort_cam_num);ind_uni=[0;ind_uni]; % find indices of begining of each camera batch
files_name=files_name(ind);
overlpped_vec=zeros(length(files_name),1); % vector of overlapped images (used to skip calculating the background if it was already calculated)
k_files=1;% initialize index running on files
kf=1;% initialize index used for rearanging the vector of filenames
%------------------------------------------------------------------

while k_files<=length(files_name) % run for every file in path
    path_name=[files_name(k_files).folder,'\',files_name(k_files).name];
    disp(path_name);
    skip=0;
    md = getCinMetaData(path_name) ;
    num_images = md.lastImage-md.firstImage+1 ;
    cindata  = myOpenCinFile(path_name);
    
%     if ~isempty(background_file_name)
%     % load a saved bg
%         bg = im2uint16(rgb2gray(imread(background_file_name)));
%     else
%         if overlpped_vec(k_files)==0 % if bg was already calculated, skip
%             first_im = myReadCinImage(cindata, md.firstImage);
%             last_im = myReadCinImage(cindata, md.lastImage);
%             
%             %roni
% %             [ bg,overlapped_bg] = CineToSparseFormat.find_bg(first_im,last_im,cindata,md);
% 
%             % omri
            bg = CineToSparseFormat.FindBGOmri(path_name);
%         end
%         bgname=sprintf('bg_cam%d.png',sort_cam_num(k_files));
%         if overlapped_bg==1 || overlpped_vec(k_files)==1 
%         % if no bg was found now (overlapped_bg) or in previous runs (overlpped_vec)
%             warning('not enough data to create background, using previous background')
%             if exist([files_name(k_files).folder,'\',bgname],'file')==2 
%             % check if a bg already exist, if so, use it
%                 bg = imread([files_name(k_files).folder,'\',bgname]);
%             else
%             % if a bg was not found reaeange the list of files so that the next cam image will be first---------
%                 if kf<k_files
%                     kf=k_files
%                 end
%                 overlpped_vec(kf+1)=1;
%                 temp=files_name(kf+1);
%                 files_name(kf+1)=files_name(ind_uni(sort_cam_num(k_files)));
%                 files_name(ind_uni(sort_cam_num(k_files)))=temp;
%                 if  kf==max(ind_uni(min(sort_cam_num(k_files)+1,max(sort_cam_num))),length(sort_cam_num)*(sort_cam_num(k_files)==max(sort_cam_num)))
%                 % send error if all files were checked and no bg was found
%                     error('no clean bg found') 
%                 end
%                 k_files=ind_uni(sort_cam_num(k_files))-1;
%                 kf=kf+1;
%                 skip=1;
%             %---------------------------------------------------------------------------------------------------
%             end
%         else
%         % if a bg was generated, save it
%             skip=0; 
%             imwrite(bg,[files_name(k_files).folder,'\',bgname]);
%         end
%     end
    
    skip=0; 

    if skip==0 % skip creating sparse movie if no bg was found or generated
        % cells only
%         sparse_array=cell(num_images+1,1);

        % struct
        sparse_array=cell(num_images,1);
        
        sparse_struct=struct;
        
        ind_fin=strfind(files_name(k_files).name, '.cine');
        save_name=files_name(k_files).name(1:ind_fin-1);
        for k = 1:num_images
            if ~mod(k,50)
                disp(k);
            end
            inp_im = myReadCinImage(cindata, md.firstImage+k-1);
            
            % old
%             mask=imbinarize(bg-inp_im,0.05);
%             mask=bwareafilt(mask,1); % choose only the largest blob
%             
%             % copy only what differs from background and save as sparse matrix
%             cropped_im=uint16(mask).*inp_im;
%             sparse_cropped=sparse(double(cropped_im));
            
            % omri
            [sparse_cropped, sigma] = CineToSparseFormat.remove_background(inp_im, bg);
            
            sparse_array{k}.sigma=sigma;
            sparse_array{k}.image=sparse_cropped;
        end
        % old
%         sparse_array{1}=bg; % first cell is background

        % omri
        sparse_struct.BG=bg;
        sparse_struct.frames=sparse_array;
        
        %old
%         save([files_name(k_files).folder,'\',save_name,'_sparse_array'],'sparse_array')

        %omri
        save([files_name(k_files).folder,'\',save_name,'sparse_struct'],'sparse_struct')
        myCloseCinFile(cindata);
%         toc
    end
    k_files=k_files+1;
end
toc
load chirp.mat;
sound(y, Fs);



% bg=sparse_struct.BG;
% for i=1:length(sparseStruct.frames)
%     im = full(sparseStruct.frames{i}.image);
%     difr=abs(double(bg)-im);
%     sigma=sparseStruct.frames{i}.sigma;
%     im(difr<=3*sigma)=0;
%     imshow(full(im),[])
%     drawnow
% %     pause(0.05)
% end
