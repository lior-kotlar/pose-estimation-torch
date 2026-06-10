function [hull_recon,real_seed] = hull_reconstruction_on_grid( all_cams,hull_params )
% Description:
% generates hull reconstruction from images loaded on all_cams object using
% hull_params as reconstruction parameters. creates only one connected
% component
% 
% Required input:
% all_cams - all_cameras_class loaded with images (each camera has 
% a field named bin_image_for_recon.Image)
% hull_params - reconstruction parametrs; hull_params_class
% 
% Output:
% hull_recon - column vector of 3D points in the hull
% real_seed - first lit voxel found

    %% initialize parametes
    queue_q = zeros(round(prod(hull_params.index_seed)/4),3);
    length_real_coord=size(hull_params.real_coord,2);
    visited = false(length_real_coord,length_real_coord,length_real_coord);
    [ ind_offsets ] = Create_offsets( hull_params.offset_index_size );
    hull_recon = NaN(size(queue_q));
    counter=1;hull_counter=1;queue_tail=0;queue_head = 0;first_ind=true;
    
    % when looking for a seed, define the indices used for searching
    % if the index is outside of the defined box erase it      
     ind_0 = (hull_params.index_seed + ind_offsets) ;
     [~,col] = find(ind_0<=0 | ind_0>=size(hull_params.real_coord,2));
     ind_0(:,col)=[];
    
     %% loop until no more neighbor voxels are lit and all voxels in queue are checked
    while (first_ind || queue_tail >= queue_head)
        % for each voxel (index) in ind_0 check if the voxel is 1 in all 3
        % images. 
        curr_index=((ind_0(:,counter))')*first_ind+abs(first_ind-1)*queue_q(queue_head+first_ind,:);
        curr_real=[hull_params.real_coord(1,curr_index(1)) hull_params.real_coord(2,curr_index(2)) hull_params.real_coord(3,curr_index(3))];
        isFly=checkVoxel(all_cams,curr_real,hull_params.all_union);
        % as long as a seed is not found look for it in every ind_0 and mark
        % this location as visited. 
        if first_ind 
            real_seed=curr_real;
            visited(curr_index(1), curr_index(2), curr_index(3)) = true ;
            first_ind=~isFly;
        end
        queue_head = queue_head + 1*(1-first_ind) ; % advance queue head by one after a seed was found
        % if the first seed was found look around it (in a distance of +-1). if
        % the voxel is 1 in all 3 cameras mark as part of the hull (hull_recon)
        if (~first_ind && isFly)    
            hull_recon(hull_counter,:) = curr_index ;
            hull_counter = hull_counter + 1 ;
            ind = uint16(curr_index' + ind_offsets(:,1:27)) ; % advance ind to a voxel +- 1 from the current one
            [~,col] = find(ind<=0 | ind>size(hull_params.real_coord,2)); % if the index is outside of the defined box erase it
            ind(:,col)=[];
            
            for p=1:size(ind,2)
                if ~visited(ind(1,p), ind(2,p), ind(3,p))
                    % if not visited, add to queue
                    queue_tail = queue_tail + 1 ; % advance queue tail if a good voxel was found and wasn't visited yet
                    queue_q(queue_tail,:) = [ind(1,p), ind(2,p), ind(3,p)] ;
                    visited(ind(1,p), ind(2,p), ind(3,p)) = true ;
                end
            end
        end
        counter=counter+1;
    end
    hull_recon(isnan(hull_recon(:,1)),:)=[]; % erase all empty voxels
    hull_recon=uint16(hull_recon);
end

function [ offset_list ] = Create_offsets(row_col_size)
% Description:
% 3D matrix of neighboors of the examined pixel, the matrix is arranged as
% a spiral; used if the original seed is dark
% 
% Required input:
% row_col_size - size of volume around seed to be searched for a lit seed 
% 
% Output:
% offset_list - 3D matrix of neighboors of the examined pixel, the matrix is arranged as
% a spiral

    len_off=row_col_size^3;
    vec_offsets=(1:row_col_size)-ceil(row_col_size/2);
    % define the movement in coordinates X,Y,Z
    ind_offsetsx=reshape(repmat(vec_offsets,row_col_size^2,1),1,len_off);
    ind_offsetsy=reshape((repmat(vec_offsets,row_col_size,row_col_size)),1,len_off);
    ind_offsetsz=repmat(vec_offsets',row_col_size^2,1)';

    ind_offsets=[ind_offsetsx;ind_offsetsy;ind_offsetsz];
    ind_offsets(:,ceil(len_off/2))=[]; % remove the [0,0,0] component
    % arange the coordinates so that the closest pixels will be located first
    [~,I]=sort(sum(ind_offsets.^2,1));
    offset_list=ind_offsets(:,I);
end

function isVoxel = checkVoxel(all_cams,voxel_coords,all_union)
% Description:
% casts the 3D point on each camera and checks if all casted pixels are lit
% 
% Required input:
% all_cams - all_cameras_class loaded with images (each camera has 
% a field named bin_image_for_recon.Image)
% voxel_coords - 3D easywand space point to check if lit (if all pixels 
% related to 3d points are lit)
% 
% Output:
% isVoxel - true if voxel is lit

    counter=0;isVoxel = false ;
    for cam_ind=1:3
        pixel_cam = dlt_inverse(all_cams.cams_array(cam_ind).dlt, voxel_coords );
        % y coordiante needs to be flipped
        pixel_cam = [round(pixel_cam(1)),round(all_cams.size_image(1)+1-pixel_cam(2))];
        if ((pixel_cam(1)>all_cams.size_image(2)) || (pixel_cam(2)>all_cams.size_image(1)) ||...
            (pixel_cam(1)<1) || (pixel_cam(2)<1))
            if ~all_union
                return
            else
                continue
            end
        end
        im=all_cams.cams_array(cam_ind).bin_image_for_recon.Image;
        counter=counter+im(pixel_cam(2),pixel_cam(1));
        if ~all_union
            if counter<cam_ind
               return;
            end
            isVoxel= (counter==3);
        else
            isVoxel= (counter>0);
        end
    end
end