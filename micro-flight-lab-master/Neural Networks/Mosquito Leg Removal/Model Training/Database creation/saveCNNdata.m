dataPath='E:\Igal\2018_09_05_igal_cutleg\with_legs_B';
sparseFilenames=extractfield(dir([dataPath,'\*.mat']),'name');
h=800;
w=1280;
% mov_name='mov2';
% mov_filenames=sparseFilenames(cell2mat(cellfun(@(x) contains(x,mov_name),...
% sparseFilenames,'UniformOutput',0)));
for k=1:numel(sparseFilenames)
    disp(k/numel(sparseFilenames));
    loady=load(fullfile(dataPath,sparseFilenames{k})); %loads sparse_array
    sparse_movie=loady.sparse_array;
    name_parts=strsplit(sparseFilenames{k},'_');
    fr=2;
    frame_count=0;
    bg=full(sparse_movie{1});
    while frame_count<16
        full_im=full(sparse_movie{fr});
        [row,col]=find(full_im);
        if (length(row)<1000)||(any(row==1|row==h))||(any(col==1|col==w))
            fr=fr+100;
            if fr>length(sparse_movie)
                break
            else
                continue
            end
        end
        
        out_inp=mat2gray(full_im);
        out_inp(full_im==0)=1;

        imwrite(out_inp,['E:/CNNdataset/images2810/',name_parts{2},...
            '_',name_parts{1},'_frame',num2str(fr),'.png']);
        
        cleaner=mat2gray(full_im).*(mat2gray(full_im)<0.85);
        
        bg_minus_im=double(bg/(2^16-1)).*double(cleaner>0)-cleaner;
        T = adaptthresh(bg_minus_im, 0.5,'NeighborhoodSize',5,'ForegroundPolarity','bright','Statistic','mean');
        mask2=imbinarize(bg_minus_im,T);
        brightest_mask=bg_minus_im>0.3;
        cleanerer=(mask2|brightest_mask).*cleaner;
        legs_mask=bwareaopen(imtophat(cleanerer>0,strel('disk',5)),10);
        out_label=double(cleanerer>0);
        out_label(legs_mask)=2;
        out_label=label2rgb(out_label,'spring','k');
        imwrite(out_label,['E:/CNNdataset/labels2clean/',name_parts{2},...
            '_',name_parts{1},'_frame',num2str(fr),'.png']);
        fr=fr+2;
        frame_count=frame_count+1;
    end
end