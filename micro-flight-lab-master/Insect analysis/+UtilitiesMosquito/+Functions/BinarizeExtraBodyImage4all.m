function BinarizeExtraBodyImage4all(allCams)
    for cam_ind=1:3
        thinner_full_mask=(mat2gray(full(allCams.cams_array(cam_ind).curr_im.imageWOBG).*allCams.cams_array(cam_ind).curr_im.body.image)>0.4)+...
            ((full(allCams.cams_array(cam_ind).curr_im.image)>0).*~allCams.cams_array(cam_ind).curr_im.body.image);
        allCams.cams_array(cam_ind).load_image(HullReconstruction.Classes.image_insect_class(sparse(thinner_full_mask).*...
            allCams.cams_array(cam_ind).curr_im.image));
    end
end