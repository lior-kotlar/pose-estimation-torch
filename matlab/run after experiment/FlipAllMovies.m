base_path="C:\Users\amita\OneDrive\Desktop\temp\next_batch_movies\";
movies = dir(fullfile(base_path));
movies = {movies.name}; 
movies_names = [];
for i=1:length(movies)
    name = movies(i);
    if contains(name, 'mov')
        movies_names = [movies_names, name];
    end
end
% for mov_ind=1:length(cam1_names)
movies = [59, 60];
% movies = length(movies_names);
for mov_ind=movies
    sparse_mov_path=fullfile(base_path, strcat('mov', string(mov_ind)), strcat('mov', string(mov_ind) ,'_cam1_sparse.mat'));
    load(sparse_mov_path)
    if ~isfield(metaData,'isFlipped')
        metaData.isFlipped=true;
    else
        metaData.isFlipped=~metaData.isFlipped;
%         metaData=rmfield(metaData,'isFlipped');
    end
    % FLIP mirrored CAM 1     
    for frame_ind=1:length(frames)
        frames(frame_ind).indIm(:,1)=(metaData(1).frameSize(1)+1)-frames(frame_ind).indIm(:,1);
    end
    % flip background
    metaData.bg=flipud(metaData.bg);
    
    save(sparse_mov_path,'frames','metaData')
    
end