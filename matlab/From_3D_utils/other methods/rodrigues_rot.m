% rodrigues_rot - Rotates array of 3D vectors by an angle theta about vector k.
% Direction is determined by the right-hand (screw) rule.
%
% Syntax:  v_rot = rodrigues(v,k,theta)
%
% Inputs:
%    v - Array of three dimensional vectors to rotate. Array can be 
%           composed of N rows of 3D row vectors or N columns of 3D column
%           vectors. If v is 3x3 array, it is assumed that it is 3 rows of
%           3 3D row vectors.
%    k - Rotation axis for each frame (does not need to be unit vector)
%    theta - Rotation angle in radians; positive according to right-hand
%           (screw) rule
%
%   Note: k and individual 3D vectors in v array must be same orientation.
%           
%
% Outputs:
%    v_rot - Array of rotated vectors.
%
% Other m-files required: dot.m (built-in MATLAB)
% Subfunctions: none
% MAT-files required: none
%
% Author: Ismail Hameduddin
%           Mechanical Engineering, Purdue University
% email: ihameduddin@gmail.com
% Website: http://www.ismailh.com
% January 2011; Last revision: 2-January-2012
%------------- BEGIN CODE --------------
function v_rot = rodrigues_rot(v,k,theta)
    [num_frames,ndims] = size(v);
    v_rot = v; % initialize rotated vector array
    num_frames = size(v, 1);
    crosskv = zeros(1,ndims); % initialize cross product k and v with right dim.
    for i = 1:num_frames
        k_i = k(i, :);
        crosskv(1) = k_i(2)*v(i,3) - k_i(3)*v(i,2);
        crosskv(2) = k_i(3)*v(i,1) - k_i(1)*v(i,3); 
        crosskv(3) = k_i(1)*v(i,2) - k_i(2)*v(i,1);
        v_rot(i,:) = cos(theta)*v(i,:) + (crosskv)*sin(theta)...
                        + k_i*(dot(k_i,v(i,:)))*(1 - cos(theta));
    end
end
%------------- END OF CODE --------------
