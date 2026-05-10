sparse_folder_path='D:\sparse_movies\2021_03_18_clean_mosquito';

info_names=dir(fullfile(sparse_folder_path,'*info.h5'));
info_names={info_names.name};

all_nose_x=[];
all_nose_y=[];
all_tail_x=[];
all_tail_y=[];

skip=93;
too_fast=20;
% for mov_ind=1:length(info_names)
for mov_ind=[1,4,5,9,11,17,18,19,29:35,45:54]
% for mov_ind=1:40

    mov_num=extractBetween(info_names{mov_ind},'mov','_');
    mov_num=mov_num{:};

    h5wi_path=fullfile(sparse_folder_path,['mov',mov_num,'_info.h5']);
    cropzone = h5read(h5wi_path,'/cropzone');

    preds_path=fullfile(sparse_folder_path,['mov',mov_num,'_body.h5']);
    preds=h5read(preds_path,'/positions_pred');
    preds = single(preds) + 1;

    outputCSVfile='wand_pts.csv';

    % nose
    node_ind=1;
    this_nose_x=squeeze(double(cropzone(2,:,:)))+squeeze(preds(node_ind+size(preds,1)/3*((1:3)-1),1,:));
    this_nose_y=squeeze(double(cropzone(1,:,:)))+squeeze(preds(node_ind+size(preds,1)/3*((1:3)-1),2,:));
    % tail
    node_ind=4;
    this_tail_x=squeeze(double(cropzone(2,:,:)))+squeeze(preds(node_ind+size(preds,1)/3*((1:3)-1),1,:));
    this_tail_y=squeeze(double(cropzone(1,:,:)))+squeeze(preds(node_ind+size(preds,1)/3*((1:3)-1),2,:));
    
    qq=diff(this_nose_x(:,1:skip:end),1,2);
    mean(qq,2)
    if any(mean(abs(qq),2)>too_fast)
        continue
    end
    all_nose_x=[all_nose_x,this_nose_x(:,1:skip:end)];
    all_nose_y=[all_nose_y,this_nose_y(:,1:skip:end)];
    all_tail_x=[all_tail_x,this_tail_x(:,1:skip:end)];
    all_tail_y=[all_tail_y,this_tail_y(:,1:skip:end)];
end

figure;hold on
plot(all_nose_x,all_nose_y,'ko')

% the format of the "points" file is:
% cam1pt1x cam1pt1y cam2pt1x cam2pt1y cam3pt1x cam3pt1y    cam1pt2x cam1pt2y cam2pt2x cam2pt2y cam3pt2x cam3pt2y

M      = zeros(size(all_tail_y,2),12);

M(:,1)  = all_nose_x(1,:);
M(:,3)  = all_nose_x(2,:);
M(:,5)  = all_nose_x(3,:);

M(:,7)  = all_tail_x(1,:);
M(:,9)  = all_tail_x(2,:);
M(:,11) = all_tail_x(3,:);

M(:,2)  = 801-all_nose_y(1,:);
M(:,4)  = 801-all_nose_y(2,:);
M(:,6)  = 801-all_nose_y(3,:);

M(:,8)  = 801-all_tail_y(1,:);
M(:,10) = 801-all_tail_y(2,:);
M(:,12) = 801-all_tail_y(3,:);

csvwrite(outputCSVfile,M) ;