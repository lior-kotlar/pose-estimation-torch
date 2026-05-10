% easy_wand_path = "G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\calibration\easywand_24_1_easyWandData.mat";
easy_wand_path = "G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\calibration\easy_wand_25_1_skip_15_outliers_removed_easyWandData.mat";
wand = load(easy_wand_path);
wand = wand.easyWandData;
wandPts = wand.wandPts;
filename = 'wandPts.csv';
writematrix(wandPts, filename);
