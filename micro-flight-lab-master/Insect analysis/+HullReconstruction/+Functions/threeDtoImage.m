function Im=threeDtoImage(dlt,threeDpts)
import HullReconstruction.Functions.dlt_inverse
% Description:
% casts 3D points to camera image given by dlt
% 
% Required input:
% dlt - 11 DLT coefficients for the camera, [11,1] array
% threeDpts - list of points to be casted
%
% Output:
% Im- casted image

    Im=false(800, 1280);
    points = round( dlt_inverse(dlt, threeDpts ));
    points(:,2)=801-points(:,2);
    Im(sub2ind(size(Im),points(:,2),points(:,1)))=1;
end