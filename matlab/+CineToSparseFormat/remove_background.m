function [fixed_image, sigma] = remove_background(img, bg)
%remove_background removes background from image
%   INPUT:
%     image - image
%     background - image of background
%   OUTPUT:
%     fixed_image - image without background

difr = abs(bg-img);

sigma = std(double(difr(:)));
thresh = 6 * sigma; %% noam changed to 6

img(difr <= thresh)=0;

fixed_image = double(img);
end % remove_background