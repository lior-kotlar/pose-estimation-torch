classdef image_mosquito_class<HullReconstruction.Classes.insect_image_class
% class containing image parameters and subimages for mosquito
    properties
        % images of wing edges (right wing indexed 1 in clusters) 
        wing_r_front_edge;
        wing_r_back_edge;
        wing_l_front_edge;
        wing_l_back_edge;
    end

    methods
        function obj=image_mosquito_class(insect_image_class_args)
        % Description:
        % Constructor 
        % 
        % Required input:
        % sparseImage - current frame grayscale image in sparse format
        %
        % Optional input:
        % CM_known - image center of mass
        %
        % Output:
        % obj- image_class
            obj@HullReconstruction.Classes.image_class(insect_image_class_args);
        end
    end 
end