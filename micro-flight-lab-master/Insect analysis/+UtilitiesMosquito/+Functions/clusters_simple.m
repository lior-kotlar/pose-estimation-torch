function cluster_pts=clusters_simple(currCoordsReal,cluster_params)
% Description:
% clusters the 3D points and arrange them by known indices -
% 1-right wing,2-left wing,3-tail,4-head,5-body
% 
% Required input:
% currCoordsReal - column vector of 3d points in the hull
% cluster_params - cluster_and_wings_params_class
%
% Output:
% cluster_pts - cell array of cluster 3d Points
% kmeans_cents_out - centroids of clusters
    
    if ~isempty(cluster_params.cents_in)
        [G,cluster_params.cents_in] = kmeans(currCoordsReal, cluster_params.num_of_clusters, 'distance','sqEuclidean', 'start',cluster_params.cents_in,'Replicates',1);
    else % if no previous centroids exist, open user ui for cluster indexing
%         T = clusterdata(currCoordsReal,'Linkage','ward','Maxclust',cluster_params.num_of_clusters);
        [G,kmeans_cents_out] = kmeans(currCoordsReal, cluster_params.num_of_clusters, 'distance','sqeuclidean', 'start','sample','Replicates',10);
        % make sure clusters are ordered 1(right),2-wings 3-tail 4-head 5-body
        color_matrix=lines(cluster_params.num_of_clusters);
        figsi=figure;
        scatter3(currCoordsReal(:,1), currCoordsReal(:,2), currCoordsReal(:,3), 36, color_matrix(G,:), 'Marker','.')
        axis equal vis3d;
        opts.Interpreter = 'tex';
        prompt = {'\color[rgb]{0,0.4470,0.7410} \clubsuit cluster 1:',...
            '\color[rgb]{0.8500,0.3250,0.0980} \clubsuit cluster 2:',...
            '\color[rgb]{0.9290,0.6940,0.1250} \clubsuit cluster 3:',...
            '\color[rgb]{0.4940,0.1840,0.5560} \clubsuit cluster 4:',...
            '\color[rgb]{0.4660,0.6740,0.1880} \clubsuit cluster 5:'};
        title = 'Input';
        dims = [1 35];
        definput = {'1','2','3','4','5'};
        answer = inputdlg(prompt,title,dims,definput,opts);
        delete(figsi)
        
        cluster_inds=str2double(answer);
        G=changem(G,cluster_inds,1:cluster_params.num_of_clusters);
        cluster_params.cents_in=kmeans_cents_out(changem(1:cluster_params.num_of_clusters,1:cluster_params.num_of_clusters,cluster_inds),:);
    end  
    
    cluster_pts=cell(cluster_params.num_of_clusters,1);
    for i=1:cluster_params.num_of_clusters
        cluster_pts{i}=currCoordsReal(G==i,:);
    end
end