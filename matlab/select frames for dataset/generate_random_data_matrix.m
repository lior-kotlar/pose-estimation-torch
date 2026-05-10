function random_data = generate_random_data_matrix(m_samples, d_featurs)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes her
mu =  3*rand(d_featurs ,1);
A = rand(d_featurs, d_featurs);
cov = A*A.' ;
random_data = mvnrnd(mu, cov, m_samples);  % multivariate random distribution
end