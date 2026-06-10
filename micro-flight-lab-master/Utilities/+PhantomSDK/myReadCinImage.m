function matlabIm = myReadCinImage (cindata, imageNo)

%{
cindata is generated in myOpenCinFile.m

cindata.cineHandle     = cineHandle ;
cindata.HRES           = HRES ;
cindata.imgSizeInBytes = imgSizeInBytes ;
cindata.imgRange       = imgRange ;
cindata.firstIm        = firstIm ;
cindata.lastIm         = lastIm ;
%}

if (imageNo<cindata.firstIm || imageNo>cindata.lastIm)
    error(['Image number must be in saved cine range [' num2str(cindata.firstIm) ';' num2str(cindata.lastIm) ']']);
end


imgRange = cindata.imgRange ;

imgRange.First = imageNo;
imgRange.Cnt   = 1;

%% Read the cine image into the buffer 
%The image will have image processings applied 
[HRES, unshiftedIm, imgHeader] = PhGetCineImage(cindata.cineHandle, imgRange, cindata.imgSizeInBytes);

%% Read image information from header
isColorImage = IsColorHeader(imgHeader);
%is16bppImage = Is16BitHeader(imgHeader);

%% Transform 1D image pixels to 1D/3D image pixels to be used with MATLAB
if (HRES >= 0)
    [unshiftedIm] = ExtractImageMatrixFromImageBuffer(unshiftedIm, imgHeader);
    if (isColorImage)
        samplespp = 3;
    else
        samplespp = 1;
    end
    bps = GetEffectiveBitsFromIH(imgHeader);
    %[matlabIm, unshiftedIm] = ConstructMatlabImage(unshiftedIm, imgHeader.biWidth, imgHeader.biHeight, samplespp, bps);
    [matlabIm, ~] = ConstructMatlabImage(unshiftedIm, imgHeader.biWidth, imgHeader.biHeight, samplespp, bps);
end

%{
%% Show image
if (showImage)
    if (isColorImage)
        figure, image(matlabIm,'CDataMapping','scaled'),colormap('default');
    else
        figure, image(matlabIm,'CDataMapping','scaled'),colormap(gray(2^8));
    end
end
%}

end

