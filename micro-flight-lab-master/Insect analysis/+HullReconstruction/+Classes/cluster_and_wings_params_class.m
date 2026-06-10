classdef cluster_and_wings_params_class<handle
% class containing cluster_and_wings parameters
    properties
        wing_flatness_thresh; % threshold for flatness (ratio betweensingular values)
        how_much_tail; % relative amount of tail to keep after cleaning
        num_of_clusters; % num_of_clusters - number of clusters
        cents_in; % cents_in - num_of_clusters of starting centroids for clusters
    end
    
    methods
        function obj=cluster_and_wings_params_class(how_much_tail,...
                wing_flatness_thresh,num_of_clusters,cents_in)
        % Description:
        % Constructor 
        % 
        % Required input:
        % how_much_tail - relative amount of tail to keep after cleaning
        % wing_flatness_thresh - threshold for flatness
        % num_of_clusters - number of clusters
        % cents_in - num_of_clusters of starting centroids for clusters
        %
        % Output:
        % obj - cluster_and_wings_params_class
        
            obj.how_much_tail=how_much_tail;
            obj.wing_flatness_thresh=wing_flatness_thresh;
            obj.num_of_clusters=num_of_clusters;
            obj.cents_in=cents_in;
        end
    end
end