function error =  compare_histograms(Data, sub_sample_indexses, bins_vector, features_weight)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
error = 0;
samples = Data(sub_sample_indexses, :);
d = length(features_weight);
nbins = length(bins_vector) - 1 ;
Data_histograms = zeros(d,nbins);

for i = 1:d
    Data_feature = Data(:, i);
    Data_histograms(i,:) = histogram(Data_feature,"NumBins",nbins,'Normalization','probability',"BinEdges", bins_vector).Values;
end

% for each feature: check distance between histograms of sample and data,
% sum the difference 
% weight according to feature weight
for i = 1:d  
    samples_feature = samples(:, i);
    hist_subsample_d =  histogram(samples_feature,"NumBins",nbins,'Normalization','probability',"BinEdges", bins_vector).Values;
    hist_Data_d = Data_histograms(i,:);    
    diff = abs(hist_Data_d - hist_subsample_d);
    distance_between_hist = sum(diff); 
    error = error + distance_between_hist * features_weight(i);
end
end

