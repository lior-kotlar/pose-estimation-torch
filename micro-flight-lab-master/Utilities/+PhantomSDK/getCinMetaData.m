function metaData = getCinMetaData(cinFilename) 
% before calling this function call:
% phantomSDK_setPath or  phantomSDK_setPath_Luca;
% LoadPhantomLibraries();
%
% a single image can be read simply by: 
% [currim, ~] = ReadCineFileImage(fileName, frameNumber, false);
% no need to use all the other crap

% get cine parameters (copied from ReadCineFileImage)
% original function is in: F:\Tsevi\MatlabCode\PhantomSDK\PhMatlabSDK\SimpleCineFileReader
% check which copy of the package needs to be used.

% metaData will contain the following fields: firstImage, lastImage,
% exists, width, height



%% Create the cine handle from the cine file.
%Is recomended that cine handle creation should be done once for a batch of image readings. 
%This will increase speed.
[HRES, cineHandle] = PhNewCineFromFile(cinFilename);
if (HRES<0)
	[message] = PhGetErrorMessage( HRES );
    error(['Cine handle creation error: ' message]);
end

metaData.exists = true ;
metaData.filename = cinFilename ;

%% read the saved range
pFirstIm = libpointer('int32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_FIRSTIMAGENO, pFirstIm);
firstIm = pFirstIm.Value;
pImCount = libpointer('uint32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_IMAGECOUNT, pImCount);
lastIm = int32(double(firstIm) + double(pImCount.Value) - 1);


metaData.firstImage = double(firstIm) ;
metaData.lastImage  = double(lastIm) ;

% get image width and height - see "Phantom SDK Reference Manual.pdf" page 99
pWidth = libpointer('int32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_IMWIDTH, pWidth);
width = pWidth.Value;

pHeight = libpointer('int32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_IMHEIGHT, pHeight);
height = pHeight.Value;

metaData.width = double(width) ;
metaData.height = double(height) ;

pframerate = libpointer('uint32Ptr',0);
PhGetCineInfo(cineHandle, PhFileConst.GCI_FRAMERATE, pframerate);
framerate = pframerate.Value;

pp = libpointer('uint32Ptr',0);
[HRES, dummyAll] = calllib('phfile','PhGetCineAuxData', cineHandle, int32(1000),uint32(1002),uint32(1002), pp);


metaData.framerate=double(framerate);

PhDestroyCine(cineHandle);

end



