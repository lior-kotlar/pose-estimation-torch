function Timemat(pathOrig)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


pathOrig= [pathOrig,'\'];
listing = dir(pathOrig);
%
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
end