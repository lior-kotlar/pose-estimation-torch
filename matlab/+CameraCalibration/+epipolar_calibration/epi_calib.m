%%
all_xs=[];
all_ys=[];
%%
sparse_folder_path='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\4cams';
% easy_wand_path='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\calibration\21_03\wand_data2_21_03_2022_skip1_easyWandData.mat';
easy_wand_path='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\calibration\21_03\ew_tst10_4.mat';

easyWandData=load(easy_wand_path);
easyWandData=easyWandData.easyWandData;
coefs=easyWandData.coefs;

mov_num='102';
frame_inds=241;
[xs,ys]=CameraCalibration.epipolar_calibration.Mark_calib_pts(...
    sparse_folder_path,easy_wand_path,mov_num,frame_inds,'num_marks',6);

ys=801-ys;

all_xs=[all_xs,xs];
all_ys=[all_ys,ys];
%%
pts=cat(3,all_xs,all_ys,ones(size(all_xs)));

f = @(dlts) CameraCalibration.epipolar_calibration.myFunc(dlts, pts) ;

% fix_cam_ind=2;
% f = @(dlt) CameraCalibration.epipolar_calibration.myFunc_1cam(dlt,fix_cam_ind,coefs, pts) ;


%10 - only cam 2
% x0 = coefs(2:end,2);
%10
x0 = coefs(2:end,:);
%11
% x0 = coefs(1:end,:);

opt2 = optimset('tolfun',1e-1) ;

[x,fval] = fminunc(f,x0,opt2);

CameraCalibration.epipolar_calibration.myFunc(x0, pts)
% CameraCalibration.epipolar_calibration.myFunc_1cam(x0,cam_ind,coefs, pts)
%% save to new easywand 1 cam
new_coefs=coefs;
for cam_ind=fix_cam_ind
    %10
    new_coefs(2:11,cam_ind)=x;
    coefs1=new_coefs(:,cam_ind);
    new_coefs(1,cam_ind)= -((coefs1(2)*coefs1(10) + coefs1(3)*coefs1(11))*(coefs1(5)*coefs1(9) + ...
    coefs1(6)*coefs1(10) + coefs1(7)*coefs1(11)) - (coefs1(2)*coefs1(6) + ...
    coefs1(3)*coefs1(7))*(coefs1(9)^2 + coefs1(10)^2 + coefs1(11)^2))/ ...
    (coefs1(9)*(coefs1(5)*coefs1(9) + coefs1(6)*coefs1(10) + coefs1(7)*coefs1(11)) ...
    - coefs1(5)*(coefs1(9)^2 + coefs1(10)^2 + coefs1(11)^2));

    [xyz,~,~,~,~,~] = CameraCalibration.epipolar_calibration.DLTcameraPosition(new_coefs(:,cam_ind));
    easyWandData.DLTtranslationVector(:,1,cam_ind)=xyz;
end
easyWandData.coefs=new_coefs;
% save('D:\sparse_movies\2022_03_06\calibration\ew_tst11.mat','easyWandData')
save('D:\sparse_movies\2022_03_06\calibration\ew_tst10.mat','easyWandData')
% save('D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\calibration\ew_tst.mat','easyWandData')
%% save to new easywand
for cam_ind=1:4
    %10
    new_coefs(2:11,cam_ind)=x(:,cam_ind);
    coefs1=new_coefs(:,cam_ind);
    new_coefs(1,cam_ind)= -((coefs1(2)*coefs1(10) + coefs1(3)*coefs1(11))*(coefs1(5)*coefs1(9) + ...
    coefs1(6)*coefs1(10) + coefs1(7)*coefs1(11)) - (coefs1(2)*coefs1(6) + ...
    coefs1(3)*coefs1(7))*(coefs1(9)^2 + coefs1(10)^2 + coefs1(11)^2))/ ...
    (coefs1(9)*(coefs1(5)*coefs1(9) + coefs1(6)*coefs1(10) + coefs1(7)*coefs1(11)) ...
    - coefs1(5)*(coefs1(9)^2 + coefs1(10)^2 + coefs1(11)^2));
    
    %11
%     new_coefs(1:11,cam_ind)=x(:,cam_ind);

    [xyz,~,~,~,~,~] = CameraCalibration.epipolar_calibration.DLTcameraPosition(new_coefs(:,cam_ind));
    easyWandData.DLTtranslationVector(:,1,cam_ind)=xyz;
end
easyWandData.coefs=new_coefs;
% save('D:\sparse_movies\2022_03_06\calibration\ew_tst11.mat','easyWandData')
% save('D:\sparse_movies\2022_03_06\calibration\ew_tst10.mat','easyWandData')
save('D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\calibration\21_03\ew_tst10_4.mat','easyWandData')
%%
easy_wand_path1='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\calibration\21_03\wand_data2_21_03_2022_skip1_easyWandData.mat';
easy_wand_path2='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\calibration\21_03\ew_tst10_3.mat';

easyWandData=load(easy_wand_path1);
easyWandData=easyWandData.easyWandData;
coefs1=easyWandData.coefs;

easyWandData=load(easy_wand_path2);
easyWandData=easyWandData.easyWandData;
coefs2=easyWandData.coefs;

cam_ind=1;
[xyz,T,ypr,Uo,Vo,Z] = CameraCalibration.epipolar_calibration.DLTcameraPosition(coefs1(:,cam_ind))
[xyz2,T2,ypr2,Uo2,Vo2,Z2] = CameraCalibration.epipolar_calibration.DLTcameraPosition(coefs2(:,cam_ind))