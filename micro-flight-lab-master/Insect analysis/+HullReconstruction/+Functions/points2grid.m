function [grid_mat,bounds]=points2grid(Coords,voxelSize)
% Description:
% takes 3D points and returns on a grid
% 
% Required input:
% Coords - 3D coordinates
% voxelSize - size of each voxel in grid
%
% Output:
% grid_mat- generated grid image
% bounds - boundaries of original hull

    % Min and max values
    bounds = [min(Coords);max(Coords)];
    % Initialize output matrix in which a non-zero entry indicates a 3D point exists in the input set
    grid_mat = zeros(round((bounds(2,1)-bounds(1,1))/voxelSize)+1,...
        round((bounds(2,2)-bounds(1,2))/voxelSize)+1,...
    round((bounds(2,3)-bounds(1,3))/voxelSize)+1,'logical');
    % For all 3D points
    for p=1:size(Coords, 1)
        i = round((Coords(p,1) - bounds(1,1))/voxelSize) + 1;
        j = round((Coords(p,2) - bounds(1,2))/voxelSize) + 1;
        k = round((Coords(p,3) - bounds(1,3))/voxelSize) + 1;
        grid_mat(i, j, k) =  1;
    end
end