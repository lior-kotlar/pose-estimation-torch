classdef hull_params_class
% class containing grid reconstruction parameters
    properties
        real_coord; % vector of the real coordinates of the grid
        real_seed; % 3d point to start the reconstruction (asstimated center of mass)
        index_seed; % 3d index to start the reconstruction on the created grid
        offset_index_size; % size of search area when original seed is not 
            % a lit voxel; 1X1 integer (90 is normal)
        all_union; % flag for full reconstruction (each camera alone)
    end
    
    methods
        function obj=hull_params_class(seed,voxelSize,volLength,offset_index_size)
        % Description:
        % Constructor 
        % 
        % Required input:
        % seed - 3d point to start the reconstruction (asstimated center of mass)
        % voxelSize - size of each voxel in meters
        % volLength  - size of the square sub-vol cube to reconstruct (meters)
        % offset_index_size - size of search area when original seed is not a lit voxel 
        %
        % Output:
        % obj - hull_params_class
        
            obj.real_seed=round(seed/voxelSize) * voxelSize; % allign to grid
            % create a vector of the real coordinates of the grid
            obj.real_coord = cell2mat(arrayfun(@(x) colon(x,voxelSize,x+...
                volLength),seed'-volLength/2,'UniformOutput',false));
            [~,I]=min(abs(obj.real_coord-obj.real_seed'),[],2);
            obj.index_seed=I;
            obj.offset_index_size=offset_index_size;
            obj.all_union=false;
        end
    end
end