sparse_folder_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\" + ...
    "datasets\one halter experiments\experiment 24-1-2024 undisturbed\arranged movies\mov15";

ew_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\" + ...
    "Utilities\Work_W_Leap\datasets\one halter experiments\experiment 24-1-2024 undisturbed\easy_wand_25_1_skip_5__easyWandData.mat";
easyWandData=load(ew_path);
easyWandData=easyWandData.easyWandData;

allCams=HullReconstruction.Classes.all_cameras_class(easyWandData);
centers=allCams.all_centers_cam';

mov_num='15';  
frame_ind=80;
[xs,ys]=Mark_calib_pts(...
    sparse_folder_path,ew_path,mov_num,frame_ind,'num_marks',4);