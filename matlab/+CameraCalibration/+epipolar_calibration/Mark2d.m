function [x,y,tags]=Mark2d(all_cams,h_fig)
%{
Description:
-----------
takes sparse movie files (triplets), crops the image and creates a data structure
compatible with the NN

Input:
-----
sparse_folder_path - path to sparse movie files.
crop_size (Optional) - size of crop window.
num_time_channels (optional name-value pair) - number of time channels for
    each frame for each camera.
time_jump (optional name-value pair) - delay between frames in the
    num_time_channels set.
numCams (optional name-value pair) - number of cameras (3?)

Output:
-----
saved_file_names - cell array of saved h5 files

Example:
-------
LegTrackerGui(sparse_folder_path,easy_wand_path)
%}

    % taking only last 3 cameras from 5234
%     clipped_fundmats=all_cams.Fundamental_matrices(:,:,4:6);
%     fund_mats=cat(3,clipped_fundmats,permute(clipped_fundmats,[2 1 3]));
%     n_cams=3;

    n_cams=length(all_cams.cams_array);
%     coupless=sortrows([nchoosek(1:n_cams,2);fliplr(nchoosek(1:n_cams,2))]);
    fund_mats=cat(3,all_cams.Fundamental_matrices,permute(all_cams.Fundamental_matrices,[2 1 3]));
% 
%     fund_mats(:,:,1:2)=all_cams.Fundamental_matrices(:,:,1:2);
%     fund_mats(:,:,3)=all_cams.Fundamental_matrices(:,:,1)';
%     fund_mats(:,:,4)=all_cams.Fundamental_matrices(:,:,3);
%     fund_mats(:,:,5)=all_cams.Fundamental_matrices(:,:,2)';
%     fund_mats(:,:,6)=all_cams.Fundamental_matrices(:,:,3)';
    
    im_heights=arrayfun(@(x) size(x.background_image,1),all_cams.cams_array);

    delete(findobj('Tag','epi_line'))
    % select points; choose cameras with leg (to stop at 2 press enter)
    [x,y,~,tags] = UtilitiesMosquito.Functions.my_ginput(n_cams,fund_mats,h_fig,im_heights);
    tags=str2num(tags)'; %#ok<ST2NM>

    if length(tags) ~= length(unique(tags))
        disp('only different axes possible!')
        pt3d=nan(3,1);
        err=nan;
        return
    end
end