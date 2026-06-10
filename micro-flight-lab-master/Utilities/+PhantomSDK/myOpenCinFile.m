function cindata = myOpenCinFile(fileName)
% based on ReadCineFileImage.m provided with the Matlab Phantom SDK
% cineHandle HRES imgSizeInBytes imgRange


%% Create the cine handle from the cine file.
%Is recomended that cine handle creation should be done once for a batch of image readings. 
%This will increase speed.
[HRES, cineHandle] = PhNewCineFromFile(fileName);
if (HRES<0)
	[message] = PhGetErrorMessage( HRES );
    error(['Cine handle creation error: ' message]);
end

%% Get information about cine
%read the saved range
pFirstIm = libpointer('int32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_FIRSTIMAGENO, pFirstIm);
firstIm = pFirstIm.Value;
pImCount = libpointer('uint32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_IMAGECOUNT, pImCount);
lastIm = int32(double(firstIm) + double(pImCount.Value) - 1);

% used when wanted to read an image
%if (imageNo<firstIm || imageNo>lastIm)
%    error(['Image number must be in saved cine range [' num2str(firstIm) ';' num2str(lastIm) ']']);
%end

%get cine image buffer size
pInfVal = libpointer('uint32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_MAXIMGSIZE, pInfVal);
imgSizeInBytes = pInfVal.Value;
%The image flip for GetCineImage function is inhibated.
pInfVal = libpointer('int32Ptr',false);
PhSetCineInfo(cineHandle, PhFileConst.GCI_VFLIPVIEWACTIVE, pInfVal);
%Create the image reange to be readed
imgRange = get(libstruct('tagIMRANGE'));

%leave the foloowing two empty (use to be "take one image at imageNo"
imgRange.First = [] ; % imageNo;
imgRange.Cnt   = [] ; % 1;

cindata.cineHandle     = cineHandle ;
cindata.HRES           = HRES ;
cindata.imgSizeInBytes = imgSizeInBytes ;
cindata.imgRange       = imgRange ;
cindata.firstIm        = firstIm ;
cindata.lastIm         = lastIm ;


end

%{
%% Read the cine image into the buffer 
%The image will have image processings applied 
[HRES, unshiftedIm, imgHeader] = PhGetCineImage(cineHandle, imgRange, imgSizeInBytes);

%% Read image information from header
isColorImage = IsColorHeader(imgHeader);
is16bppImage = Is16BitHeader(imgHeader);

%% Transform 1D image pixels to 1D/3D image pixels to be used with MATLAB
if (HRES >= 0)
    [unshiftedIm] = ExtractImageMatrixFromImageBuffer(unshiftedIm, imgHeader);
    if (isColorImage)
        samplespp = 3;
    else
        samplespp = 1;
    end
    bps = GetEffectiveBitsFromIH(imgHeader);
    [matlabIm, unshiftedIm] = ConstructMatlabImage(unshiftedIm, imgHeader.biWidth, imgHeader.biHeight, samplespp, bps);
end

%% Show image
if (showImage)
    if (isColorImage)
        figure, image(matlabIm,'CDataMapping','scaled'),colormap('default');
    else
        figure, image(matlabIm,'CDataMapping','scaled'),colormap(gray(2^8));
    end
end


end
%}