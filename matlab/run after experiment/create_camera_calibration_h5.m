% create camera calibration h5

clearvars -except easy_wand_path savePath
addpath '/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/matlab/From_3D_utils'
addpath '/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/matlab/run after experiment/get_camera_matrix_decomposition'
addpath '/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/micro-flight-lab-master/Insect analysis'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% change for each experiment (or override from CLI / wrapper script)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('easy_wand_path','var') || isempty(easy_wand_path)
    easy_wand_path = "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/inference_datasets/new_roni_experiments/2022_02_03/calibration/1+2_03_02_2022_skip5_dltCoefs_V2_easyWandData.mat";
end
if ~exist('savePath','var') || isempty(savePath)
    savePath = "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/inference_datasets/new_roni_experiments/2022_02_03/calibration/calibration.h5";
end
fprintf('easy_wand_path = %s\nsavePath = %s\n', easy_wand_path, savePath);

easyWandData=load(easy_wand_path);
% wandPnts = easyWandData.easyWandData.wandPts;
% csvwrite('wandPnts.csv', wandPnts);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);

%%
rotation_matrix = allCams.Rotation_Matrix;
camera_centers = allCams.all_centers_cam';
inv_camera_matrices = zeros(4,4,3);
camera_matrices = zeros(4,3,4);

rotation_matrices = zeros(4, 3, 3);
K_matrices = zeros(4, 3, 3);
translations = zeros(4, 3);

for cam = 1:4
    inv_camera_matrices(cam, :, :) = allCams.cams_array(cam).invDLT;
    camera_matrices(cam, :, :) = allCams.cams_array(cam).reshaped_dlt;

    % get camera parameters
    M = squeeze(camera_matrices(cam, :, :));
    [K, Rc_w, Pc, pp, pv] = DecomposeCamera(M);
    
    M_prime = K * [Rc_w, -Rc_w * Pc];

    % checks
    t = M(:, end);
    t_prime = inv(K) * t;
    % M_prime = K * [Rc_w, t_prime];
    cs = camera_centers(cam, :);
    cs_prime = -Rc_w' * t_prime;
    err1 = mean(abs(cs_prime - cs'));
    err2 = mean(abs(M_prime(:) - M(:)));

    rotation_matrices(cam, :, :) = Rc_w;
    K_matrices(cam, :, :) = K;
    translations(cam, :) = t_prime;
end




%%
h5create(savePath, '/rotation_matrix', size(rotation_matrix));
h5create(savePath, '/camera_centers', size(camera_centers));
h5create(savePath, '/inv_camera_matrices', size(inv_camera_matrices))
h5create(savePath, '/camera_matrices', size(camera_matrices));


h5create(savePath, '/rotation_matrices', size(rotation_matrices));
h5create(savePath, '/K_matrices', size(K_matrices))
h5create(savePath, '/translations', size(translations));

% Write data to each dataset in the HDF5 file
h5write(savePath, '/rotation_matrix', rotation_matrix);
h5write(savePath, '/camera_centers', camera_centers);
h5write(savePath, '/inv_camera_matrices', inv_camera_matrices);
h5write(savePath, '/camera_matrices', camera_matrices);


h5write(savePath, '/rotation_matrices', rotation_matrices);
h5write(savePath, '/K_matrices', K_matrices);
h5write(savePath, '/translations', translations);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rotation_matrix = allCams.Rotation_Matrix;
% camera_centers = allCams.all_centers_cam';
% inv_camera_matrices = zeros(4,4,3);
% camera_matrices = zeros(4,3,4);
% for cam = 1:4
%     inv_camera_matrices(cam, :, :) = allCams.cams_array(cam).invDLT;
%     camera_matrices(cam, :, :) = allCams.cams_array(cam).reshaped_dlt;
% end
% 
% M1 = squeeze(camera_matrices(1, :, :));
% 
% % [K, Rc_w, Pc, pp, pv] = DecomposeCamera(M1);
% % t = M1(:, end);
% % t_prime = inv(K) * t;
% % M1_prime = K * [Rc_w, t_prime];
% 
% %%
% h5create(savePath, '/rotation_matrix', size(rotation_matrix));
% h5create(savePath, '/camera_centers', size(camera_centers));
% h5create(savePath, '/inv_camera_matrices', size(inv_camera_matrices))
% h5create(savePath, '/camera_matrices', size(camera_matrices));
% 
% % Write data to each dataset in the HDF5 file
% h5write(savePath, '/rotation_matrix', rotation_matrix);
% h5write(savePath, '/camera_centers', camera_centers);
% h5write(savePath, '/inv_camera_matrices', inv_camera_matrices);
% h5write(savePath, '/camera_matrices', camera_matrices);




