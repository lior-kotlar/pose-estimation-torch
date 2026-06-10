classdef hull3d_class<handle
% class containing hull parameters and subhulls
    properties
        CM; % hull center of mass
        hull; % 3d points array
        principalVectors; %
        principalValues; %
    end

    methods
        function obj=hull3d_class(hullPoints)
        % Description:
        % Constructor 
        % 
        % Required input:
        % hullPoints - list of 3d points
        %
        % Output:
        % obj- hull3d_class
        
            obj.hull=hullPoints;
            % find the blob's center of mass
            rCM=mean(hullPoints);
            if isnan(rCM)
                disp('¡¡¡Warning: hull is empty!!!')
                obj.CM=[0,0,0];
            else
                obj.CM=rCM;
            end
            [~,s,ev]=svd(hullPoints-rCM,0);
            obj.principalVectors=ev;
            obj.principalValues=s;
        end
        
        function imageOut=hull2image(obj,dlt)
            imageOut=HullReconstruction.Functions.threeDtoImage(dlt,obj.hull);
        end
    end
end