classdef all_cameras_class<handle&matlab.mixin.Copyable
% class containing all cameras in scene (all cameras from easyWand)

    properties
        cams_array; % array of cameras (camera_class)
        size_image; % size of the camera output image
        Rotation_Matrix; % rotation matrix given by alligning cameras directions sum to [0,0,1]
        Fundamental_matrices; % (3*3*number of cameracouples) array with fundamental matrix ordered F12,F13,..
        currCM3D; % the 3D point of intersection of centers of mass from 2D loaded images
        all_centers_cam; % matrix of all camera centers in easyWand space (column vectors)
        all_DLT_coefs; % matrix of all DLT coefficients (column vectors)
        hull_params; % class containing grid reconstruction parameters
        currHull3d;
    end
    
    methods
        function obj=all_cameras_class(easyWandData)
        % Description:
        % Constructor 
        % 
        % Required input:
        % easyWandData - easywand struct
        % 
        % Output:
        % obj- all_cameras_class related to easyWandData
        
            obj.all_DLT_coefs=easyWandData.coefs;
            obj.size_image=[easyWandData.imageHeight(1),easyWandData.imageWidth(1)];
            obj.cams_array = HullReconstruction.Classes.camera_class.empty(easyWandData.nCams,0);
            obj.all_centers_cam=squeeze(easyWandData.DLTtranslationVector);
            % add all cameras to array and calculate sum of camera
            % directions
            cameras_up=zeros(3,1);
            for cam_ind=1:easyWandData.nCams
                obj.cams_array(cam_ind)=HullReconstruction.Classes.camera_class(easyWandData,cam_ind);
                cameras_up=cameras_up+obj.cams_array(cam_ind).camera_dir;
            end
            obj.Rotation_Matrix=vrrotvec2mat(vrrotvec(cameras_up,[0,0,1]));
            obj.Fundamental_matrices=HullReconstruction.Functions.get_fundamental_mats(obj);
        end
        
        function [hull_recon,real_seed] = hull_reconstruction_on_grid(obj,field_string,...
                voxelSize,volLength,offset_index_size)
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
            obj.load_ims_for_reconstruction(field_string);
            hull_parameters=HullReconstruction.Classes.hull_params_class(obj.currCM3D,voxelSize,volLength,offset_index_size);            
            obj.hull_params=hull_parameters;
            queue_q = zeros(round(prod(hull_parameters.index_seed)/4),3);
            length_real_coord=size(hull_parameters.real_coord,2);
            visited = false(length_real_coord,length_real_coord,length_real_coord);
            [ ind_offsets ] = HullReconstruction.Functions.Create_offsets( hull_parameters.offset_index_size );
            hull_recon = NaN(size(queue_q));
            counter=1;hull_counter=1;queue_tail=0;queue_head = 0;first_ind=true;

            % when looking for a seed, define the indices used for searching
            % if the index is outside of the defined box erase it      
             ind_0 = (hull_parameters.index_seed + ind_offsets) ;
             [~,col] = find(ind_0<=0 | ind_0>=size(hull_parameters.real_coord,2));
             ind_0(:,col)=[];

             %% loop until no more neighbor voxels are lit and all voxels in queue are checked
            while (first_ind || queue_tail >= queue_head)
                % for each voxel (index) in ind_0 check if the voxel is 1 in all 3
                % images. 
                curr_index=((ind_0(:,counter))')*first_ind+abs(first_ind-1)*queue_q(queue_head+first_ind,:);
                curr_real=[hull_parameters.real_coord(1,curr_index(1)) hull_parameters.real_coord(2,curr_index(2)) hull_parameters.real_coord(3,curr_index(3))];
                isFly=obj.checkVoxel(curr_real,hull_parameters.all_union);
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
                    [~,col] = find(ind<=0 | ind>size(hull_parameters.real_coord,2)); % if the index is outside of the defined box erase it
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
        
        function load_ims_for_reconstruction(obj,field_string)
        % Description:
        % assigns the image to be reconstructed & update 3D center of mass
        % 
        % Required input:
        % obj - all_cameras_class
        % field_string - name of field to be loaded from image_clas
            
            PB=zeros(3,4);
            for cam_ind=1:length(obj.cams_array)
%                 im_to_load=obj.cams_array(cam_ind).curr_im.(field_string);
                if iscell(field_string)
                    spltstr=strsplit(field_string{cam_ind},'.');
                else
                    spltstr=strsplit(field_string,'.');
                end
                im_to_load=getfield(obj.cams_array(cam_ind).curr_im,spltstr{:});
                if contains(class(im_to_load),'image','IgnoreCase',true)
                    im_to_load=im_to_load.image;
                end
                if ~islogical(im_to_load) % in case image is not binary (should be sparse)
                    im_to_load=full(im_to_load)>0;
                end
                obj.cams_array(cam_ind).bin_image_for_recon=HullReconstruction.Classes.image_class(im_to_load);
                
                centerPt=[obj.cams_array(cam_ind).bin_image_for_recon.CM(1);...
                    obj.size_image(1)+1-obj.cams_array(cam_ind).bin_image_for_recon.CM(2);1];
                PB(cam_ind,:)=obj.cams_array(cam_ind).invDLT*centerPt;
            end
            ends=PB(:,1:3)./(PB(:,4));
            obj.currCM3D=HullReconstruction.Functions.lineIntersect3D(obj.all_centers_cam',ends);
        end
        
        function isVoxel = checkVoxel(obj,voxel_coords,all_union)
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
                pixel_cam = HullReconstruction.Functions.dlt_inverse(obj.cams_array(cam_ind).dlt, voxel_coords );
                % y coordiante needs to be flipped
                pixel_cam = [round(pixel_cam(1)),round(obj.size_image(1)+1-pixel_cam(2))];
                if ((pixel_cam(1)>obj.size_image(2)) || (pixel_cam(2)>obj.size_image(1)) ||...
                    (pixel_cam(1)<1) || (pixel_cam(2)<1))
                    if ~all_union
                        return
                    else
                        continue
                    end
                end
                im=obj.cams_array(cam_ind).bin_image_for_recon.image;
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
        
        function CastClusters(obj)
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
            for cam_ind=1:length(obj.cams_array)
                
                obj.cams_array(cam_ind).curr_im.wingRight=HullReconstruction.Classes.image_wing_class(obj.CastCleanMosquito('wingRight',cam_ind));
                obj.cams_array(cam_ind).curr_im.wingLeft=HullReconstruction.Classes.image_wing_class(obj.CastCleanMosquito('wingLeft',cam_ind));
                obj.cams_array(cam_ind).curr_im.body=HullReconstruction.Classes.image_body_class(obj.CastCleanMosquito('body',cam_ind));
                obj.cams_array(cam_ind).curr_im.body.tail=HullReconstruction.Classes.image_class(obj.CastCleanMosquito('body.tail',cam_ind));
                obj.cams_array(cam_ind).curr_im.body.head=HullReconstruction.Classes.image_class(obj.CastCleanMosquito('body.head',cam_ind));
                obj.cams_array(cam_ind).curr_im.body.torso=HullReconstruction.Classes.image_class(obj.CastCleanMosquito('body.torso',cam_ind));
            end
        end
        
        function cleanImage=CastCleanMosquito(obj,fieldString,cam_ind)
            
            spltstr=strsplit(fieldString,'.');
            hullToCast=getfield(obj.currHull3d,spltstr{:});
            cleanImage=bwareafilt(imclose(hullToCast.hull2image(obj.all_DLT_coefs(:,cam_ind)),strel('disk',3))&...
                            full(obj.cams_array(cam_ind).curr_im.image),1);
        end
    end
end

