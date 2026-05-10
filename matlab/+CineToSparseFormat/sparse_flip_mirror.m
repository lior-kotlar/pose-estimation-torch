base_path='G:\My Drive\Amitai\Bees + UV + magnetic field 14-9\movies + README\mov2';
cam1_names=dir(fullfile(base_path,'*cam1*.mat'));
cam1_names={cam1_names.name};
% for mov_ind=1:length(cam1_names)
for mov_ind=1:size(cam1_names,1)
    sparse_mov_path=fullfile(base_path,cam1_names{mov_ind});
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
    
    % add fields to other cameras
%     for other_cam=2:4
%         other_sparse_mov_path=replace(sparse_mov_path,'cam5',['cam',num2str(other_cam)]);
%         load(other_sparse_mov_path,'metaData')
%         if ~isfield(metaData,'isFlipped')
%             metaData.isFlipped=false;
%         else
%             continue
%         end
%         save(other_sparse_mov_path,'metaData','-append')
%     end
end