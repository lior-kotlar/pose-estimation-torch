clear
close all
clc

load hull_mov83
hull.body.angles % contains pitch roll and yaw

figure;
ax1 = subplot(3,1,1);hull.plotAngle('pitch','body',1,'disterb',7.5,'time',0,'interp',1);
ax2 = subplot(3,1,2);hull.plotAngle('roll','body',1,'disterb',7.5,'time',0,'interp',1);
ax3 = subplot(3,1,3);hull.plotAngle('yaw','body',1,'disterb',7.5,'time',0);
linkaxes([ax1,ax2,ax3],'x');

%%
hull_frame_good=1000;

sparse_frame_good=hull_frame_good+hull.general.VideoData.sparseFrame-1;
%%

% body_x=hull.body.vectors.X; 

% why 3 dimensions?
% why angles have different num of frames
% body_x=hull.body.vectors.X;
% body_x=hull.body.vectors.X;


body_CM=hull.body.hullAndBound.CM;

hull.rightwing.angles.phi;
hull.rightwing.angles.theta;
hull.rightwing.angles.psiA;

body_pitch=hull.body.angles.roll;
body_pitch=hull.body.angles.pitch;
body_pitch=hull.body.angles.yaw;

