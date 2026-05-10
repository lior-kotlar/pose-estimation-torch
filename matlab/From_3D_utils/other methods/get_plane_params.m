function plane_P = get_plane_params(pnts_3D)
%GET_PLANE_PARAMS Summary of this function goes here
%   Detailed explanation goes here
[coeff,score,latent] = pca(pnts_3D);
normal = coeff(:,3);
d = -normal'*mean(pnts_3D,1)'; % The distance from origin to the plane is -dot(normal,mean)
plane_P = [normal', d];
end

