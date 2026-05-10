clear
close all
clc


% pathOrig='H:\My Drive\Ronis Exp\sagiv\22_11_30_1600-end\hull\';
pathOrig='H:\My Drive\dark 2022\2023_08_10_100ms\hull\\';
pathOrig= [pathOrig,'\'];
name4Folder = strsplit(pathOrig,'\');
path2=[pathOrig,name4Folder{end-1},'_Reorder\','\'];
listing = dir(pathOrig);

createTime_meat = 1

camera1_ofset = 0
camera2_ofset = 0
camera3_ofset = 0
camera4_ofset = 0

minofset_cam1 = 0
secofset_cam1 = 0
milisecofset_cam1 = 0
hourofset_cam1 = 0;

minofset_cam3 = 0
secofset_cam3 = -32
milisecofset_cam3 = 52
hourofset_cam3 = 0;




if createTime_meat == 1
    Timemat(pathOrig);
end
load([pathOrig,'\','time'],'time');
%% arange files in folder so each folder movie will contain sparse files with the same trigger time. 
camname = {'cam1','cam2','cam3','cam4'};
mkdir(path2);
four_camflg = 1;
    timesK = strings(size(time,1),1)
for k = 1:1:size(time,1) 
   date =  split(time{k,2});
   timesK(k,1) = string([date{2},' ',date{3},' ',time{k,3}]);
end
[v ind] = sortrows(timesK,'ascend');
timeReorder = [time(:,1),time(:,3:end)];

% create the directories, use the sorted time array to create them
warning off
for k=1:1:size(timeReorder,1)
    movdir=sprintf('mov%d\\',timeReorder{ind(k),1});
    mkdir([path2,movdir]);
    movname=sprintf('mov%d_cam1_sparse.mat',timeReorder{ind(k),1});
    if isempty(timeReorder{ind(k),2}) ~= 1
     copyfile([pathOrig,movdir,movname],[path2,movdir,movname],'f');
    end
    if isempty(timeReorder{ind(k),2}) ~= 1
        tm =  str2double(regexp(timeReorder{ind(k),2}, '\d+', 'match'));
        if camera1_ofset == 1
        tm = timeofset(tm,hourofset_cam1,minofset_cam1,secofset_cam1,milisecofset_cam1);
        end
        cam1time(k,:) = tm;
        movcam1{k,1} =movdir; 
        movcam1{k,2} =timeReorder{ind(k),1}; 
    end
end
%% move the files of camera 2 3 4 to the folders created for camera 1, compare the times and make sure they are the same
for kcam = 3:5
    cmstr = sprintf('Camera # %d',kcam)
    display(cmstr)
for k=1:1:size(timeReorder,1) 
if isempty(timeReorder{ind(k),kcam}) ~= 1
    
        
    tm = str2double(regexp(timeReorder{ind(k),kcam}, '\d+', 'match'));
    if kcam == 5 && camera4_ofset == 1
        tm = timeofset(tm,hourofset_cam4,minofset_cam4,secofset_cam4,milisecofset_cam4);   
    end
    if kcam == 4 && camera3_ofset == 1
        tm = timeofset(tm,hourofset_cam3,minofset_cam3,secofset_cam3,milisecofset_cam3);   
    end
    if kcam == 3 && camera2_ofset == 1
        tm = timeofset(tm,hourofset_cam2,minofset_cam2,secofset_cam2,milisecofset_cam2);   
    end
    
    tm = tm(1:4);
    [~,indint] = intersect(cam1time(:,1:4),tm,'rows');
    
    movdir_orig = sprintf('mov%d\\',timeReorder{ind(k),1});
    movname_orig = sprintf('mov%d_cam%d_sparse.mat',timeReorder{ind(k),1},kcam - 1);
    movnamenew=sprintf('mov%d_cam%d_sparse.mat',movcam1{indint,2},kcam - 1);
    if length(indint) == 1
        copyfile([pathOrig,movdir_orig,movname_orig],[path2,movcam1{indint,1},movnamenew],'f')
    end
end
end
end

