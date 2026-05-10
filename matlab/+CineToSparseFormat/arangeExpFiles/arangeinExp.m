function arangeinExp(path)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
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

end