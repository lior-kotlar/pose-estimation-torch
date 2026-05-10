clear
close all
clc


path='H:\My Drive\dark 2022\2023_08_10_100ms\hull\'
listing = dir(path);
for k=1:1:length(listing)
    movfile=strfind(listing(k).name,'mov');
    file_end=strfind(listing(k).name,'.m');
    if isempty(movfile)==0 && isempty(file_end)==0
        name_splt=split(listing(k).name,'_');
        if exist([path,name_splt{1}],'dir')==0
            mkdir([path,name_splt{1}])
        end
        moveFrom=[path,listing(k).name];
        moveto=[path,name_splt{1}]
        movefile(moveFrom,moveto);
    end
end
