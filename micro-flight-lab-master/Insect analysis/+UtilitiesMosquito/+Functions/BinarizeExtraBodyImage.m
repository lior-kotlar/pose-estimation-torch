function imageOut=BinarizeExtraBodyImage(imageIn)
%     imageOut=
    thinner_full_mask=(mat2gray(full(allCams.cams_array(cam_ind).curr_im.ImageWOBG).*body_casted)>0.4)+...
        ((full(allCams.cams_array(cam_ind).curr_im.Image)>0).*~body_casted);
    mosquito_image_class(sparse(thinner_full_mask).*...
        allCams.cams_array(cam_ind).curr_im.Image);
end