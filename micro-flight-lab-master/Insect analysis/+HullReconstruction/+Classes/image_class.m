classdef image_class<handle
% class containing image parameters and subimages
    properties
        CM; % image center of mass; 1X2 integer vector
        image; % original grayscale image in sparse format
        imageWOBG; % image after subtracting the background
    end

    methods
        function obj=image_class(sparseImage,CM_known)
        % Description:
        % Constructor 
        % 
        % Required input:
        % sparseImage - current frame grayscale image in sparse format
        %
        % Optional input:
        % CM_known - image center of mass; 1X2 integer vector
        %
        % Output:
        % obj- image_class
        
            obj.image=sparseImage;
            % find the blob's center of mass
            if nargin==1
                [y,x]=find(full(sparseImage));
                if isempty(x)
                    disp('¡¡¡Warning: image is empty!!!')
                    obj.CM=[1,1];
                else
                    obj.CM=round([mean(x),mean(y)]);
                end
            elseif nargin==2
                obj.CM=CM_known;
            end
        end
    end 
end