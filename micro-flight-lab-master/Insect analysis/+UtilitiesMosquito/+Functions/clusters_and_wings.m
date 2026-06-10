function [cluster_pts,evecs]=clusters_and_wings(currCoordsReal,cluster_params)
% Description:
% clusters the 3D points, fixes tail and wings and
% 
% Required input:
% currCoordsReal - column vector of 3d points in the hull
% cluster_params - cluster_and_wings_params_class
%
% Output:
% cluster_pts - cell array of cluster 3d Points
% evecs - principal components of clusters
% kmeans_cents_out - centroids of clusters
    
    %% initialize variables
    cluster_pts=cell(cluster_params.num_of_clusters,1);
    evecs=cell(cluster_params.num_of_clusters,1);
    %% kmeans clustering
    [G,cluster_params.cents_in] = kmeans(currCoordsReal, cluster_params.num_of_clusters, 'distance','sqEuclidean', 'start',cluster_params.cents_in,'Replicates',1);
    %% break to clusters and clean if wing/tail
    body_center=cluster_params.cents_in(5,:);
    
    for clust_ind=1:cluster_params.num_of_clusters
        cluster_pts{clust_ind}=currCoordsReal(G==clust_ind,:);
%         full1=pcdenoise(pointCloud(currCoordsReal(G==clust_ind,:)),'NumNeighbors',30,'Threshold',0.5);
%         cluster_pts{clust_ind}=full1.Location;
        
        % remove excess unconnected parts
        cluster_pts{clust_ind} = make_hull_connected(cluster_pts{clust_ind});
        
        if clust_ind==1||clust_ind==2 % wings 
            [~,s,evecs{clust_ind}]=svd(cluster_pts{clust_ind}-cluster_params.cents_in(clust_ind,:),0); %center the data
            flatness=s(1,1)/s(3,3);
            if flatness<cluster_params.wing_flatness_thresh % wing is too fat, something's wrong
                disp('wing is fat!!!')
%                 cluster_pts{clust_ind} = make_thin_wing(cluster_pts{clust_ind},body_center);
%                 [~,s,~]=svd(cluster_pts{clust_ind}-mean(cluster_pts{clust_ind}),0); %center the data
%                 if (s(1,1)/s(3,3))<cluster_params.wing_flatness_thresh
%                 keyboard
%                 end
            end
        elseif clust_ind==3 % tail
            cluster_pts{clust_ind} = make_tail_tight(cluster_pts{clust_ind},...
                body_center,cluster_params.how_much_tail);            
        else % body&head
        end
        [~,~,evecs{clust_ind}]=svd(cluster_pts{clust_ind}-mean(cluster_pts{clust_ind}),0); %center the data  
        %% set main components to point outwards
        if clust_ind~=5
            if dot(mean(cluster_pts{clust_ind})-body_center,evecs{clust_ind}(:,1))<0
                evecs{clust_ind}(:,1)=-evecs{clust_ind}(:,1);
            end
        end
    end
end