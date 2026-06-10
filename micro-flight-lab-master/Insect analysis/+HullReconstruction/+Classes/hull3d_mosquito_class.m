classdef hull3d_mosquito_class<HullReconstruction.Classes.hull3d_insect_class
% class containing hull parameters and subhulls
    properties
        
    end

    methods
        
        function obj=hull3d_mosquito_class(hull3d_insect_class_args)
        % Description:
        % Constructor 
        % 
        % Required input:
        % hullPoints - list of 3d points
        %
        % Output:
        % obj- hull3d_class
            obj@HullReconstruction.Classes.hull3d_insect_class(hull3d_insect_class_args);
        end
        
        function clusters_simple(obj,cluster_params)
        % Description:
        % Constructor 
        % 
        % Required input:
        % hullPoints - list of 3d points
        %
        % Output:
        % obj- hull3d_class
        clusters=UtilitiesMosquito.Functions.clusters_simple(obj.hull,cluster_params);
%         cluster_list={'wingRight','wingLeft','body.tail','body.head','body.torso'};
        obj.wingRight=HullReconstruction.Classes.hull3d_wing_class(clusters{1});
        obj.wingLeft=HullReconstruction.Classes.hull3d_wing_class(clusters{2});
        obj.body=HullReconstruction.Classes.hull3d_body_class(cell2mat(clusters(3:5)));
        
        obj.body.tail=HullReconstruction.Classes.hull3d_class(clusters{3});
        obj.body.head=HullReconstruction.Classes.hull3d_class(clusters{4});
        obj.body.torso=HullReconstruction.Classes.hull3d_class(clusters{5});
        end
        
        function clusters_and_wings(obj,cluster_params)
        % Description:
        % Constructor 
        % 
        % Required input:
        % hullPoints - list of 3d points
        %
        % Output:
        % obj- hull3d_class
        clusters=UtilitiesMosquito.Functions.clusters_and_wings(obj.hull,cluster_params);
%         cluster_list={'wingRight','wingLeft','body.tail','body.head','body.torso'};
        obj.wingRight=HullReconstruction.Classes.hull3d_wing_class(clusters{1});
        obj.wingLeft=HullReconstruction.Classes.hull3d_wing_class(clusters{2});
        obj.body=HullReconstruction.Classes.hull3d_body_class(cell2mat(clusters(3:5)));
        
        obj.body.tail=HullReconstruction.Classes.hull3d_class(clusters{3});
        obj.body.head=HullReconstruction.Classes.hull3d_class(clusters{4});
        obj.body.torso=HullReconstruction.Classes.hull3d_class(clusters{5});
        end
    end 
end