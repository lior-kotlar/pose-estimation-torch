function [Data, best_frames, worst_frames, min_error, max_error] = SampleBestFramesDark2022(num_of_frames_to_sample, iterations, num_of_His_bins, features_weights)
%SAMPLEBESTFRAMESDARK2022 Summary of this function goes here
%   Detailed explanation goes here
% move 16 good
% mov 7: 1 - 3150
% mov 10: 1 - 4800
% mov 12 fly enters cam 4 start from 1200
% mov 14: 1 - 3000
% mov 15: 1 - 4300
% mov 17: 1 - 5000
% mov 18: 230 - untill end

num_of_movies = 18;
preff = 'C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder';
num_of_features = 4;
Data = zeros(0,num_of_features);
for i = 1 : num_of_movies
    if i == 9 || i == 18
        continue
    end    
    hull_addr = strcat(preff,'\mov', string(i), '\hull_op\hull_Smov',  string(i), '.mat');
    load(hull_addr);
    if i == 7 
        num = 3150;
    elseif i == 10
        num = 4800;
    elseif i == 14
        num = 3000;
    elseif i == 15
        num = 4300;
    elseif i == 17
        num = 5000;
    else
        num = length(SHull.rightwing.angles.phi);
    end
    start = 1;
    if i == 18
        start = 230;
    end
    phi_i = SHull.rightwing.angles.phi(start: num);
    yaw_i = SHull.body.angles.yaw(start: num);
    pitch_i = real(SHull.body.angles.pitch(start: num));
    roll_i = SHull.body.angles.roll(start: num);
    movie_num = ones(num - start + 1, 1) * i;
    offset = SHull.general.VideoData.sparseFrame - 1; 
    index_in_movie = (start:num) + offset;
    Data = [Data ; [phi_i yaw_i pitch_i roll_i, movie_num, index_in_movie']];
end 
Data(Data(:,1) > 200 , :) = [];  % delete rows of phi bigger then 200
[best_frames, worst_frames, min_error, max_error] = sample_from_data(Data, num_of_frames_to_sample, iterations, num_of_His_bins, features_weights);
end

