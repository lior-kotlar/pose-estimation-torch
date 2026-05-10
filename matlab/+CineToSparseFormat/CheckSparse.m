%% 
% Load PhantomSDK if needed: (!!take care: PHANTOM SDK LOADING NEEDS TO 
% BE REVISED GLOBALLY!!)

if ~exist('getCinMetaData','file')
    run('PhantomSDK\runMeFirst.m')
end
%%
main_path='D:\mosquito_audio2_11042021';
cine_names=dir(fullfile(main_path,'*.cine'));
cine_names={cine_names.name};

skip=13;
for cine_ind=1:length(cine_names)
    disp(cine_ind)
    cine_path=fullfile(main_path,cine_names{cine_ind});
    sparse_path=[cine_path(1:(end-5)),'.mat'];


    cine_metadata = getCinMetaData(cine_path) ;
    cine_data  = myOpenCinFile(cine_path);



    num_images_cine = cine_metadata.lastImage-cine_metadata.firstImage+1 ;
    dummy_im=myReadCinImage(cine_data, cine_metadata.firstImage);
    max_val=intmax(class(dummy_im));
    min_val=intmin(class(dummy_im));
    diff_norm=double(max_val)*numel(dummy_im);

    sparse_struct=load(sparse_path);
    num_images_sparse=size(sparse_struct.frames,1);

    assert(num_images_sparse==num_images_cine,'Number of frames differ!!!')

    frame_ind=0;
    fprintf('\n');
    line_length = fprintf('frame: %u/%u',frame_ind,num_images_cine);
    for frame_ind=1:skip:num_images_sparse
        fprintf(repmat('\b',1,line_length))
        line_length = fprintf('frame: %u/%u',frame_ind,num_images_cine);

        cine_im = myReadCinImage(cine_data, cine_metadata.firstImage+frame_ind-1);

        sparse_im=sparse_struct.metaData.bg;
        frame=sparse_struct.frames(frame_ind);

        sparse_im(sub2ind(sparse_struct.metaData.frameSize,frame.indIm(:,1),frame.indIm(:,2)))=frame.indIm(:,3);

        diff_im=abs(double(sparse_im)-double(cine_im));

    %     if length(frame.indIm)>50
        big_diffs(frame_ind,cine_ind)=sum(diff_im(:)>(max_val/9));
    %     else
    %         big_diffs(frame_ind)=nan;
    %     end


    %     if ~mod(frame_ind,30)
    %         imshow(diff_im,[min_val,max_val])
    %     imagesc(diff_im)
    %     drawnow
    %         plot(diff_im(:))
    %         histogram(diff_im(:),10)
    %     end
    %     imshow(cine_im)
    %     sum_diffs(frame_ind)=mean(diff_im(:));
    %     max_diffs(frame_ind)=max(diff_im(:));
    end
end
%%
fly_area=2000;
txt_displace=fly_area/20;
h =figure;

ax1=subplot(2,1,1);
hold on
yline(fly_area,'--')
text(0,fly_area-txt_displace,'full fly')
yline(fly_area/2,'--')
text(0,fly_area/2-txt_displace,'1/2 fly')
yline(fly_area/10,'--')
text(0,fly_area/10-txt_displace,'1/10 fly')

hPlot = plot(big_diffs,'.');

xlabel('frame no.')
ylabel('difference [no. of pixels]')
title('difference between cine and sparse frames')

[max_val,max_ind]=max(big_diffs);
cursorMode = datacursormode(h);
hDatatip = cursorMode.createDatatip(hPlot);
pos = [max_ind,big_diffs(max_ind),0];
set(hDatatip, 'Position', pos)         
updateDataCursors(cursorMode)

ax2=subplot(2,1,2);
plot(cellfun(@length,{sparse_struct.frames.indIm}))
xlabel('frame no.')
ylabel('size of sparse object [no. of pixels]')
title('size of sparse frames')


linkaxes([ax1,ax2],'x')