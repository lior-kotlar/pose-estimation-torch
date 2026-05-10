function [all_fly4bb,bin_im] = bin4overlap(bg,im)

% zeros_im=zeros(size(bg));
% bin_im=imbinarize(bg,'adaptive','ForegroundPolarity','bright','Sensitivity',0.4);

bin_im=imbinarize(bg);

all_fly4bb=imreconstruct(bin_im,~imbinarize(im,'adaptive','ForegroundPolarity','dark'));
% [row,col]=find(all_fly4bb>0);
% zeros_im(min(row):max(row),min(col):max(col))=~imbinarize(im(min(row):max(row),min(col):max(col)));

end