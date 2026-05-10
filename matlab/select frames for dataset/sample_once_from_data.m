function sub_sample_of_indexes = sample_once_from_data(Data, num_samples)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

last_frames = 100;


m = length(Data) - last_frames;
sub_sample_of_indexes = randperm(m, num_samples);  % sample random indexes
% sub_sample = Data(random_indexes, :);  % sample random rows
end
