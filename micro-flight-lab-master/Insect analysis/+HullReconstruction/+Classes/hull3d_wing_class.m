classdef hull3d_wing_class<HullReconstruction.Classes.hull3d_class
% class containing hull parameters and subhulls
    properties
        leadingEdge; % body 3d hull
        trailingEdge; % body 3d hull
    end

    methods
        function obj=hull3d_wing_class(hull3d_class_args)
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