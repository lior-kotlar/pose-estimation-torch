function [best_subsample, worse_subsample, min_error, max_error] = sample_from_data(Data, num_of_samples, iterations, bins_vector, features_weight)

% sample num_of_samples indexes of samples in Data, such that they
% represent The data as good as we could find using 'iterations' iterations. 
% each iteration we peak at random n indexes and check how much it
% represent the data and take the best sample.
% we check it via compering the histogram of each column to the Data's histogram for this column
% at the end we take the sample with the minimum comparison error. 

% iterations and num_of_bins are hyperparameters


[m,d] = size(Data);
all_samples = zeros(iterations, num_of_samples);
for i = 1 : iterations
    sub_sample_indexses = sample_once_from_data(Data, num_of_samples);
    all_samples(i, :) = sub_sample_indexses;
end

% check the distance between the sample and the real data histograms
errors = zeros(iterations,1);
% find best sample 
for i = 1 : iterations
    sub_sample_indexses = all_samples(i, :);
    error_i = compare_histograms(Data, sub_sample_indexses, bins_vector, features_weight);  
    errors(i) = error_i; 
    if mod( i , 100 ) == 0
        disp(i);
    end
end
[M, min_idx] = min(errors);
[M, max_idx] = max(errors);
min_error = errors(min_idx);
max_error = errors(max_idx);
% return best and worse samples 
best_subsample = all_samples(min_idx, :);
worse_subsample = all_samples(max_idx, :);


