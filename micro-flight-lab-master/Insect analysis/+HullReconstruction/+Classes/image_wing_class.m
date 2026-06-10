classdef image_wing_class<HullReconstruction.Classes.image_class
% class containing i	age parameters and subimages for mosquito
    properties
        trailingEdge;
        leadingEdge;
        boundary;
    end

    methods
        function obj=image_wing_class(image_class_args)
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
            obj@HullReconstruction.Classes.image_class(image_class_args);
        end
    end 
end