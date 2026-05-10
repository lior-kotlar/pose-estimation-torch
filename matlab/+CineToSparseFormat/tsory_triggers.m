if ~exist('getCinMetaData','file')
    run('PhantomSDK\runMeFirst.m')
end
%%
% UI to get a list of filenames and initialize variables:
[fileNames,folderPath] = uigetfile('*.cine','Select Cine\s to convert',...
    'MultiSelect', 'on');
if isequal(fileNames,0)
   disp('User selected Cancel; end of session')
   return
end
if ~iscell(fileNames)
    fileNames={fileNames};
end
fileInd=0;
%%
while fileInd<length(fileNames)
    fileInd=fileInd+1;
%% 
    cinePath=fullfile(folderPath,fileNames{fileInd});
    xmlStruct(fileInd)=getXML(cinePath);
    trig_t{fileInd}=xmlStruct(fileInd).CineFileHeader.TriggerTime.Time.Text;
    serial{fileInd}=xmlStruct(fileInd).CameraSetup.Serial;
end
%%
% trig_t=xmlStruct(1).CineFileHeader.TriggerTime.Time.Text;
% trig_h=str2double(trig_t(1:2))
% trig_h=str2double(trig_t(1:2))
% xmlStruct(1).CameraSetup.Serial

function xmlStruct=getXML(cinePath)
    cineData  = myOpenCinFile(cinePath);
    cineMetaData = getCinMetaData(cinePath) ;
    PhLVRegisterClientEx([], PhConConst.PHCONHEADERVERSION);
    PhSetUseCase(cineData.cineHandle, PhFileConst.UC_SAVE);
    % set to save xml
    pXML = libpointer('bool',true);
    PhSetCineInfo(cineData.cineHandle,PhFileConst.GCI_SAVEXML,pXML);
    % Prepare a save name and path for the cine
    tempFileName='C:\just4xml\just4xml';
    pInfVal = libpointer('cstring', [tempFileName,'.cine']);
    PhSetCineInfo(cineData.cineHandle, PhFileConst.GCI_SAVEFILENAME, pInfVal);
    % save only first frame
    imgRng = get(libstruct('tagIMRANGE'));
    imgRng.First = cineMetaData.firstImage;
    imgRng.Cnt = 1;
    pimgRng = libpointer('tagIMRANGE', imgRng);
    PhSetCineInfo(cineData.cineHandle, PhFileConst.GCI_SAVERANGE, pimgRng);
    % save cine and xml
    PhWriteCineFile(cineData.cineHandle);
    % load xml to matlab and delete temporary files
    xmlStruct=CineToSparseFormat.betterXml2struct([tempFileName,'.xml']).chd;
    delete([tempFileName,'.cine'])
    delete([tempFileName,'.xml'])
end