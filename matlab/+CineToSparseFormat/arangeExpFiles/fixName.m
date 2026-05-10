close all
clc
clear


pathorig = 'H:\My Drive\dark 2022\2023_08_10_100ms\';
path_rename = 'H:\My Drive\dark 2022\2023_08_10_100ms\hull\';
mkdir(path_rename)
% Get all text files in the current folder
files = dir([pathorig]);


% Loop through each file 
for cam = 1:1:4
    movnum = 1;
for id = 1:length(files)
    cammane = sprintf('cam%d',cam);
    isCamfile = strfind(files(id).name,cammane);
    if isCamfile ~= 0
        % Get the file name 
       
    [~, f,ext] = fileparts(files(id).name);
    
      rename = split(f,'_') ; 
      filerename = sprintf('mov%d_%s_sparse.mat',movnum,cammane);
      
      copyfile([pathorig,files(id).name], [path_rename,filerename]); 
      movnum = movnum + 1;
        
    end
end
end

% arangeinExp(path_rename)
