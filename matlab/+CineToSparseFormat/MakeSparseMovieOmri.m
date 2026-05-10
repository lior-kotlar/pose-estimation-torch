tic
% user input:
folderPath='E:\barTest\more'; % path of video files

% get list of names of files and initialize------------------------
fileNames = dir([folderPath,'\*.cine']);
fileNames = {fileNames.name};
fileInd=0;% initialize index running on files
%------------------------------------------------------------------

while fileInd<length(fileNames) % run for every file in path
    fileInd=fileInd+1;
%     fileInd=3;
    cinePath=fullfile(folderPath,fileNames{fileInd});
    disp(cinePath);
    
    metaData = getCinMetaData(cinePath) ;
    numOfImages = metaData.lastImage-metaData.firstImage+1 ;
    cineData  = myOpenCinFile(cinePath);
    
    %%
    bg = CineToSparseFormat.FindBGOmri(cinePath);
    checkFig=figure;
    imshow(bg,[0,2^16-1])
    answer = questdlg('Is background good?', ...
        'Background Check', ...
        'Yes','No','Yes');
    delete(checkFig)
    % Handle response
    switch answer
        case 'Yes'
            
        case 'No'
            continue
        otherwise
            continue
    end
    
    
    %%
    savePath= fullfile(folderPath,...
        [fileNames{fileInd}(1:(strfind(fileNames{fileInd}, '.cine')-1)),'_sparse']);
    
    sparseArray=cell(numOfImages,1);
    [height,width]=size(bg);
    sparseStruct=struct;
    
    saveInd=0;
    for imageInd = 1:numOfImages
%     for imageInd = 1:20:numOfImages %created for body angle sampling!!!
        saveInd=saveInd+1;
        if ~mod(imageInd,50)
            disp(imageInd);
        end
        inp_im = myReadCinImage(cineData, metaData.firstImage+imageInd-1);
%         [croppedImage, sigma] = CineToSparseFormat.remove_background(inp_im, bg);
        
%         old
        
%         initial_mask=zeros(size(inp_im));
%         initial_mask(360:560,450:700)=1;
%         [L,Centers] = imsegkmeans(inp_im,2);
%         B = labeloverlay(inp_im,L);
%         
%         imshow(activecontour(bg-inp_im,initial_mask,1000,'Chan-Vese').*...
%             double(inp_im),[])
%         
%         T = adaptthresh(bg-inp_im,0.01,'ForegroundPolarity','bright');
%         imshow(imbinarize(bg-inp_im,T));
%         drawnow
% 
        mask=imbinarize(bg-inp_im,0.05);
%         mask=bwareafilt(mask,1); % choose only the largest blob
        mask=bwareaopen(mask,50); % made for multiple insects in frame
% %         mask=bwareaopen(mask,1000);
% 
%         % copy only what differs from background and save as sparse matrix
        cropped_im=uint16(mask).*inp_im;
        croppedImage=double(cropped_im);
%         imshow(croppedImage,[])
%         drawnow
%         sparseArray{imageInd}.sigma=sigma;
        sparseArray{saveInd}.image=sparse(croppedImage);
    end

    sparseStruct.bg=bg;
    sparseStruct.startFrame=metaData.firstImage;
    sparseStruct.frames=sparseArray;
    sparseStruct.frameRate=metaData.framerate;
    
    save(savePath,'sparseStruct')
    myCloseCinFile(cineData);
end

toc
load chirp.mat;
sound(y, Fs);
