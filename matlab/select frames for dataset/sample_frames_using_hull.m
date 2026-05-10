function [Data, best_frames, worst_frames, min_error, max_error] = sample_frames_using_hull(num_of_frames_to_sample, iterations, num_of_His_bins)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

load hull_mov83; 
% m * 3 matrix of x,y,z of CM
body_CM = hull.body.hullAndBound.CM;

% important angle of wing
hull_rightwing_phi = hull.rightwing.angles.phi;
hull_rightwing_theta = hull.rightwing.angles.theta;
hull_rightwing_psiA = hull.rightwing.angles.psiA;

body_roll = hull.body.angles.roll;
body_pitch = hull.body.angles.pitch;
body_yaw = hull.body.angles.yaw;

num_of_featurs = 1;
m = length(body_yaw);
Data = zeros(m, num_of_featurs);
Data(:, 1) = hull_rightwing_phi;
% Data(:, 1:3) = body_CM;
% Data(:, 4) = hull_rightwing_phi;
% Data(:, 5) = hull_rightwing_theta;
% Data(:, 6) = hull_rightwing_psiA;
% Data(:, 7) = body_roll;
% Data(:, 8) = real(body_pitch);
% Data(:, 9) = body_yaw;

% weight of each feature in decision
% features_weight = [0 0 0 4 0 0 0 0 0];
n = find(Data==0, 1, 'last');
%  Data(155:m, :)
features_weight = [1];
[best_frames, worst_frames, min_error, max_error] = sample_from_data(Data(n:m, :), num_of_frames_to_sample, iterations, num_of_His_bins, features_weight);
best_frames = best_frames + n;
end



