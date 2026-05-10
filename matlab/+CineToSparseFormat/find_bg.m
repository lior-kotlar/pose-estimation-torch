function [ bg,overlapped_bg ] = find_bg( im1,im2,cindata,md )
% Find background
% need to add case - if the fly in the last frame is on top the fly in the
% first frame
overlapped_bg=0;
k=1;

% substract 2 images (usually the first and last of the movie) and generate their binary images
[bin_bg_im2,bin_im2] = CineToSparseFormat.bin4overlap(im1-im2,im2);
[bin_bg_im1,bin_im1] = CineToSparseFormat.bin4overlap(im2-im1,im1);

% dilate both images
SE = strel('disk',5);
BW2 = uint16(imdilate(bin_bg_im2,SE)*1);
BW1 = uint16(imdilate(bin_bg_im1,SE)*1);

% check if the dilated images overlap
overlap=sum(sum((BW2+uint16(BW1))==2));
ind_overlap=find((BW2+uint16(BW1))==2);

im_fly=BW2.*im1; % multiply the mask of image 1 with image 2 to get the BG at the location of the insect
bg=im2.*(1-BW2)+im_fly; % create the BG by adding missing image im_fly


% in case of overlap, define the first frame as the middle of the movie
% and count the overlapping pixels

numOfFrame=(md.lastImage+md.firstImage)/2;
imoverlap=zeros(800,1280);
imoverlap(ind_overlap)=1; % create an mask of the overlapped pixels


while overlap>0
    inp_im = myReadCinImage(cindata,round((numOfFrame))); % load a new frame
    % create a binary image of the new frame and dilate it
    [bin_bg_inp_im,bin_im2] = CineToSparseFormat.bin4overlap(im1-inp_im,inp_im); 
    SE = strel('disk',5);
    bin_bg_inp_im = imdilate(bin_bg_inp_im,SE)*1;
    
    
    imoverlap_new=(bin_bg_inp_im*1+imoverlap)==2; % create an image of the area overlapping the new image and the overlap mask
    add2bg=inp_im.*uint16(-imoverlap_new+imoverlap); % create an image from the non overlapping pixels (-imoverlap_new+imoverlap)
    bg=bg.*(1-uint16(-imoverlap_new+imoverlap))+add2bg;
    imoverlap=imoverlap.*imoverlap_new; % generate the new overlaping image
    if sum(imoverlap(:))==0
        overlap=0 % the bg is ready, exit the function 
    else
        numOfFrame=numOfFrame+k;
        
        if round(numOfFrame)==md.lastImage % if we reached the end of the movie go back to the middle and examine the previus frames
            numOfFrame=round((md.lastImage+md.firstImage)/2);
            k=-1;
        end
        if round(numOfFrame)==md.firstImage % the bg has overlapping pixels, exit the function
            overlapped_bg=1;
            sum(imoverlap(:))
            break
        end
    end
    
    
end




end