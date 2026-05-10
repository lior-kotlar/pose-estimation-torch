clear
close all
clc


pathOrig='H:\My Drive\dark 2022\2022_03_03\hull\';
pathOrig= [pathOrig,'\'];
name4Folder = strsplit(pathOrig,'\');
path2=[pathOrig,name4Folder{end-1},'_Reorder\','\'];
listing = dir(pathOrig);
%% create time matrix - a matrix that holds the trigger time for each movie
if exist([pathOrig,'time.mat'],'file') == 0
        disp('creating mat0rix of trigger time')
    for k=1:1:length(listing)
        movfile=strfind(listing(k).name,'mov');
        movNum = str2double(regexp(listing(k).name, '\d+', 'match'));
        
        if isempty(movfile)==0
            dirName=listing(k).name;
            listing_file = dir([pathOrig,'/',dirName,'/']);
            
            for kcam=1:1:length(listing_file)
                
                camfile=strfind(listing_file(kcam).name,'cam');
                if isempty(camfile)==0
                    load([pathOrig,'/',dirName,'/',listing_file(kcam).name]);
                    numCam=str2num(listing_file(kcam).name(camfile+3));
                    time{movNum,1}=movNum;
                    if strfind(metaData.xmlStruct.CineFileHeader.TriggerTime.Date.Text,'2000')~=0
                        ind = strfind(metaData.xmlStruct.CineFileHeader.TriggerTime.Date.Text,'2000')
                        metaData.xmlStruct.CineFileHeader.TriggerTime.Date.Text(ind:end) = '2022' 
                    end
                        
                    time{movNum,2}=metaData.xmlStruct.CineFileHeader.TriggerTime.Date.Text;

                    time{movNum,numCam+2}=[metaData.xmlStruct.CineFileHeader.TriggerTime.Time.Text];
                      
                    
                  
                    
                end
                
                
            end
        end
    end
    save([pathOrig,'\','time'],'time')
else
    load([pathOrig,'\','time'],'time')
end
% %% arange files in folder so each folder movie will contain sparse files with the same trigger time. 
% camname = {'cam1','cam2','cam3','cam4'};
% mkdir(path2);
% four_camflg = 1;
%     timesK = strings(size(time,1),1)
% for k = 1:1:size(time,1)
%     
%    date =  split(time{k,2});
% timesK(k,1) = string([date{2},' ',date{3},' ',time{k,3}]);
% 
% 
% end
% [v ind] = sortrows(timesK,'ascend');
% timeReorder = [time(:,1),time(:,3:end)];
% 
% for k=1:1:size(timeReorder,1)
%     movdir=sprintf('mov%d\\',timeReorder{ind(k),1});
%     movdirnew = sprintf('mov%d\\',timeReorder{k,1});
%     movcount = sprintf('mov %d / %d',timeReorder{k,1},size(timeReorder,1));
%     disp(movcount)
% 
%     if exist([path2,movdirnew],'dir')==0
%     mkdir([path2,movdirnew]);
%     end
% 
%     listing = dir([pathOrig,movdir]);
%     
%     for k4cam = 1:1:length(listing) % check if 3 or 4 cameras
%         if strfind(listing(k4cam).name,'cam5') >0
%             four_camflg = 1;
%         end
%     end
%     
%     try
%         movname=sprintf('mov%d_cam4_sparse.mat',timeReorder{ind(k),1});
%         movnamenew=sprintf('mov%d_cam4_sparse.mat',timeReorder{k,1});
% 
%         copyfile([pathOrig,movdir,movname],[path2,movdirnew,movnamenew],'f')
%         
%         for kcam = 2:1:4+four_camflg
%             camTime{kcam-1} = str2double(regexp(timeReorder{ind(k),kcam}, '\d+', 'match'));
%         end
%         
%         for kcam = 1:1:length(camTime)   
%             keepMoveFile_time(time,camname{kcam},ind(k),pathOrig,movdir,path2,camTime,kcam,k,movdirnew);
%         end  
%     catch
%         continue
%     end
%     
% end
% 
