clear
close all
clc

currpath = pwd;
pth = strsplit(currpath,'\');

addpath('G:\Documents\micro-flight-lab\Utilities')


for mov=[71]
     counter=0;  
    try
    parhOfSparses='G:\My Drive\Amitai\experiment magnet + UV 30.8\movies\'; % 2023_08_09_60ms and 2023_08_09_80ms
    movdir=sprintf('mov%d\\',mov);
    flName=dir([parhOfSparses,movdir])
    for k=1:1:4
        movie{k}=sprintf('mov%d_cam%d_sparse.mat',mov,k)
    end
    for k=1:1:length(flName)
        file_2change=flName(k).name;  
        include_par=contains(file_2change,'partition')
        [camnum]=strfind(file_2change,'cam')
        counter=include_par+1;
    
        if counter>3
            error('check partition name')
        elseif include_par==1
             movname=sprintf('mov%d_cam%d_sparse.mat',mov,str2num(file_2change(camnum+3)))
         
            movefile([parhOfSparses,movdir,file_2change],[parhOfSparses,movdir,movname],'f');
        end
    end
    
    
    
    
    fileNames={[parhOfSparses,movdir,movie{1}];[parhOfSparses,movdir,movie{2}];[parhOfSparses,movdir,movie{3}];[parhOfSparses,movdir,movie{4}]};
    
    VideoEditing.Join4Sparses(fileNames,30,25,'skip',1)
    catch
        continue
    end
end