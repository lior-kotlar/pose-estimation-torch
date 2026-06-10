classdef hull3d_body_class<HullReconstruction.Classes.hull3d_class
% class containing hull parameters and subhulls
    properties
        head; % body 3d hull
        torso; % body 3d hull
        tail; % body 3d hull
    end

    methods
        function obj=hull3d_body_class(hull3d_class_args)
        % Description:
        % Constructor 
        % 
        % Required input:
        % hullPoints - list of 3d points
        %
        % Output:
        % obj- hull3d_class
            obj@HullReconstruction.Classes.hull3d_class(hull3d_class_args);
        end
    end 
end