function bg = FindCineBG(cinePath,varargin)
%{
Description:
-----------
- Scans a single cine and generates a background image.

Input:
-----
cinePath - cine path.
nFrames (Optional) - number of frames to sample across the cine.
method (Optional) - method of choosing background pixels.

Example:
-------
bg = FindCineBG('xx.cine','Single movie max')
%}
%% parse inputs and initialize variables
    inpParser = inputParser;
    addRequired(inpParser,'fileNames');
    addOptional(inpParser,'nFrames',100);
    addOptional(inpParser,'method','Single movie max',@(x)ischar(x));
    parse(inpParser,cinePath,varargin{:});
    
    cineData = myOpenCinFile(cinePath);
    cineMetaData = getCinMetaData(cinePath);
    nFrames = min([double(cineMetaData.lastImage)-double(cineMetaData.firstImage)+1,...
        inpParser.Results.nFrames]);
    counter=0;

    mov = zeros([nFrames,size(myReadCinImage(cineData, cineMetaData.firstImage))],...
        class(myReadCinImage(cineData, cineMetaData.firstImage)));
    
    frameInds=round(linspace(1,double(cineMetaData.lastImage)...
            -double(cineMetaData.firstImage)+1,nFrames));
    meanVals=zeros(size(frameInds));
%% loop on frames to get the sample
    for frameInd = frameInds
        counter=counter+1;
        inpIm=myReadCinImage(cineData, cineMetaData.firstImage+frameInd-1);
        mov(counter,:,:)=inpIm;
        meanVals(counter)=mean(inpIm(:));
    end
    myCloseCinFile(cineData);
%% obtain background using method
    switch inpParser.Results.method
        case 'Single movie max better'
            bg=squeeze(max(mov));
            %% better bg using non insect values
            [~,maxInd]=max(meanVals);
            inpIm=squeeze(mov(maxInd,:,:));
            mask=imbinarize(bg-inpIm,0.05);
            mask=bwareaopen(mask,50);
            newBg=inpIm;
            remInds = find(mask);    
            remCounter=0;
            while ~isempty(remInds)
                remCounter=remCounter+1;
                inpIm=squeeze(mov(remCounter,:,:));
                mask=imbinarize(bg-inpIm,0.05);
                mask=bwareaopen(mask,50); % made for multiple insects in frame
                newBGinds=setdiff(remInds,intersect(remInds,find(mask)));
                remInds=intersect(remInds,find(mask));
                newBg(newBGinds)=inpIm(newBGinds);
            end
            bg=newBg;
        case 'Single movie max'
            bg=squeeze(max(mov));
        case 'Single movie mean'
            bg=squeeze(mean(mov));
        case 'Single movie median'
            bg=squeeze(median(mov));
    end
end