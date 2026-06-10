classdef camera_class<handle
% class containing camera parameters and loaded image of current frame
    properties
        dlt; % DLT matrix
        cam_number; % number of camera
        camera_cnt; % camera center in easyWand space
        camera_dir; % camera direction in easyWand space
        reshaped_dlt; % reshaped DLT (3 X 4 matrix)
        invDLT; % psuedo inverse of DLT matrix
        curr_im; % image_class of current frame
        prev_im; % image_class of previous frame
        bin_image_for_recon; % image to be used during reconstruction
        background_image; % background image (taken from sparse_move{1})
    end
    
    methods
        function obj=camera_class(easyWandData,cam_ind)
        % Description:
        % Constructor 
        % 
        % Required input:
        % easyWandData - easywand struct
        % cam_ind - index of camera in easyWandData struct
        %
        % Output:
        % obj- camera_class related to cam_ind in easyWandData 
        
            obj.cam_number=cam_ind+1;
            obj.camera_cnt=easyWandData.DLTtranslationVector(:,:,cam_ind);
            obj.dlt=easyWandData.coefs(:,cam_ind);
            obj.reshaped_dlt=reshape([easyWandData.coefs(:,cam_ind);1],[4,3])';
            obj.invDLT=pinv(obj.reshaped_dlt);
            % calculate a 3D point on the casted ray using inverse DLT
            PB=obj.invDLT*[easyWandData.imageWidth(1)/2;easyWandData.imageHeight(1)/2;1];
            % camera direction needs to be normalized from homogenous coordinates
            cam_dir=PB(1:3)/PB(4)-obj.camera_cnt; 
            obj.camera_dir=cam_dir/norm(cam_dir);        
        end
        
        function load_image(obj,curr_image)
        % Description:
        % loads the current frame image (as image_class) on the camera 
        % 
        % Required input:
        % obj - camera_class
        % curr_im - image_class of current frame
            obj.curr_im=curr_image;
            full_im=full(obj.curr_im.image);
            if ~isempty(obj.background_image)
                obj.curr_im.imageWOBG=sparse(mat2gray(obj.background_image.*(full_im>0)-full_im));
            end
        end
        
        function load_background(obj,background_image)
        % Description:
        % loads a background image on the camera 
        % 
        % Required input:
        % obj - camera_class
        % background_image - uint16 background image

            obj.background_image=double(background_image);
        end
    end
end