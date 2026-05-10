classdef sparseMovieClass
% class containing image parameters and subimages
    properties
        frames; % sparse matrix of entire movie(1280*(800*number Of Frames))
        bg; % background image (uint16)
        startFrame; % cine frame number of the first frame
    end

    methods
        function obj=sparseMovieClass()
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

        end
        
        
    end 
end