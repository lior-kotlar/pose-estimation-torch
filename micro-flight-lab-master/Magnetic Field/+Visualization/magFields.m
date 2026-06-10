%% Description:
% Magnetic field visualzation/calculation
%% set directories and path
tic
%home
% addpath(genpath('C:\Users\Noamler\Documents\git reps\micro_flight_lab\Insect analysis'));
% dataPath='G:\My Drive\Micro Flight Group\Videos\2018_08_29_igal_cutleg';
% small_cage_path='C:\Users\Noamler\Documents\git reps\micro_flight_lab\Magnetic Field\miniCage_center.stl';
%lab
addpath(genpath('C:\git reps\micro_flight_lab\Insect analysis'));
dataPath='C:\google drive\Micro Flight Group\Videos\2018_08_29_igal_cutleg';
small_cage_path='C:\git reps\micro_flight_lab\Magnetic Field\miniCage_center.stl';
%% load easywand data and create the all_cameras_class
easyWandData=load([dataPath,'\29_08_2018_easyWandData.mat']);
allCams=all_cameras_class(easyWandData.easyWandData);
%% load white images
for cam_ind=1:3
    allCams.cams_array(cam_ind).load_image(image_class(ones(800,1280)));
end
%% set parameters and initialize variables
plot_flag=true; 
save_flag=false;

% initialize variables
%% check if plot or save data/images/video
if plot_flag
    figi=figure('color','w','units','normalized',...
               'outerposition',[0 0 1 1]);
    axi=axes('XColor','none','YColor','none','ZColor','none');
    hold on
    axis equal vis3d manual
    grid off
end
if save_flag
    save_dir='figs\wo1\';
    if ~exist(save_dir,'dir')
        mkdir(save_dir)
    end
end
%% clear irrelevant variables
clear('loady','sparseFilenames','easyWandData','frames_per_flap',...
    'how_much_tail','wing_flatness_thresh','num_of_clusters','cents_in')
%% create hull of 3D frame
% parameters for grid reconstruction
voxelSize = 1e-3 ; % size of grid voxel
volLength= 8e-2 ; % size of the square sub-vol cube to reconstruct (meters)
offset_index_size=90; % size of search area when original seed is not a true voxel

allCams.load_ims_for_reconstruction('Image'); % updates images and seed for reconstruction
[hull_params]=hull_params_class(allCams.currCM3D,voxelSize,volLength,offset_index_size);
[hull_inds,~] = hull_reconstruction_on_grid( allCams,hull_params );
% translate to easywand space; rotate to lab frame
hull_3dfr=[hull_params.real_coord(1,hull_inds(:,1))',hull_params.real_coord(2,hull_inds(:,2))',...
    hull_params.real_coord(3,hull_inds(:,3))']*allCams.Rotation_Matrix';
%% create hull of all camera frames (full reconstruction)
voxelSize = 10e-3 ; % size of grid voxel
volLength= 50e-2 ; % size of the square sub-vol cube to reconstruct (meters)
hull_params.all_union=true;

allCams.load_ims_for_reconstruction('Image'); % updates images and seed for reconstruction
[hull_params]=hull_params_class(allCams.currCM3D,voxelSize,volLength,offset_index_size);
hull_params.all_union=true;
[hull_inds,~] = hull_reconstruction_on_grid( allCams,hull_params );
% translate to easywand space; rotate to lab frame
hull_full_recon=[hull_params.real_coord(1,hull_inds(:,1))',hull_params.real_coord(2,hull_inds(:,2))',...
    hull_params.real_coord(3,hull_inds(:,3))']*allCams.Rotation_Matrix';
%% plot hull
if plot_flag
    % generate cage from stl 
    
    [cage_x,cage_y,cage_z,cage_c] = stlread(small_cage_path);
    % fix to m units
    patch(0.001*cage_x,0.001*cage_y,0.001*cage_z,0.001*cage_c); 
    
    pcshow(hull_3dfr,'r')
    pcshow(hull_full_recon,'k','MarkerSize',20)

%     view(c/n_frames*45*num_flaps,36)
    title('Full Setup')
    if save_flag
        savefig(figi,[save_dir,num2str(fr)],'compact')
        writeVideo(outputVideo, getframe(figi));
        cla(axi)
    end
end
%% calculate & visualize magnetic field - vertical
% define field params
mu0=(4*pi)*10^-7;
rho_copper= 1.68e-8;

a=0.095; % ring radius
% a=0.055;
sol_len=0.02;
N=170; %number of turns
% N=240; %number of turns

I_per_turn=5;
Itot=I_per_turn*N;
I_dens=Itot/sol_len;

defl_coils=0.05; % half-distance between coils in helmholtz
wire_length=2*pi*a*N;
wire_rad=1e-3;

R_coil=rho_copper*wire_length/(pi*(wire_rad)^2);
R_vert=R_coil;
V_out=I_per_turn*R_coil;
P_out=V_out*I_per_turn;

% tau=L/R_coil;
% L=0.12;
% 0.5*mu0*I/a % ring
%% magnet parameters
% rho= 1.68e-8;
% B_set=1e-2;
% N_set=100;
% V_set=32;
% P_max=200;
% 
% V=P_max/2*mu0*N_set/(B_set*a);
% wire_rad=sqrt(B_set*a^2*4*rho/(mu0*V));
% R_coil=rho*2*a*N_set/wire_rad^2;
% 
% I=V/R_coil;
% 
% coil_width=sqrt(N_set)*2*wire_rad;
%%
num_pix=40;
r=linspace(0,1.1*a,num_pix);
z=linspace(0,0.1,num_pix);
[r_grid,z_grid] = meshgrid(r,z);
%%
coil_flag='solenoid';
plane_flag='yz';
% calculate

switch coil_flag
    case 'solenoid'
        zeta_p=z_grid+sol_len/2;
        zeta_m=z_grid-sol_len/2;
        alph=@(zet) a./sqrt(zet.^2+(a+r_grid).^2);
        bet=@(zet) zet./sqrt(zet.^2+(a+r_grid).^2);
        k=@(zet) sqrt((zet.^2+(a-r_grid).^2)./(zet.^2+(a+r_grid).^2));
        gamm=(a-r_grid)./(a+r_grid);

        Br=mu0*I_dens/pi*(alph(zeta_p).*...
            gcei(k(zeta_p),ones(size(z_grid)),1,-ones(size(z_grid)))-...
            alph(zeta_m).*gcei(k(zeta_m),ones(size(z_grid)),1,-ones(size(z_grid))));
        
        Bz=mu0*I_dens/pi*(a./(a+r_grid)).*(bet(zeta_p).*...
            gcei(k(zeta_p),gamm.^2,1,gamm)-...
            bet(zeta_m).*gcei(k(zeta_m),gamm.^2,1,gamm));
            
        k0=(sol_len/2)/sqrt((sol_len/2)^2+a^2);
        L=8/3*mu0*(N/sol_len*a)^2*(sqrt((sol_len/2)^2+a^2)*gcei(k0,1,1,2*k0^2)-a);
        L_vert=L;
    case 'ring'
        k=4*r_grid*a./(z_grid.^2+(a+r_grid).^2);
        [E1,E2] = ellipke(k);
        Br=mu0*Itot*z_grid./(2*pi*r_grid.*sqrt(z_grid.^2+(a+r_grid).^2)).*...
            ((z_grid.^2+r_grid.^2+a^2)./(z_grid.^2+(r_grid-a).^2).*E2-E1);
        Bz=mu0*Itot./(2*pi*sqrt(z_grid.^2+(a+r_grid).^2)).*...
            ((a^2-z_grid.^2-r_grid.^2)./(z_grid.^2+(r_grid-a).^2).*E2+E1);
    case 'helm'
        k_up=4*r_grid*a./((z_grid+defl_coils).^2+(a+r_grid).^2);
        [E1_up,E2_up] = ellipke(k_up);
        k_down=4*r_grid*a./((z_grid-defl_coils).^2+(a+r_grid).^2);
        [E1_down,E2_down] = ellipke(k_down);
        Br_up=mu0*Itot*(z_grid+defl_coils)./(2*pi*r_grid.*sqrt((z_grid+defl_coils).^2+(a+r_grid).^2)).*...
            (((z_grid+defl_coils).^2+r_grid.^2+a^2)./((z_grid+defl_coils).^2+(r_grid-a).^2).*E2_up-E1_up);
        Br_down=mu0*Itot*(z_grid-defl_coils)./(2*pi*r_grid.*sqrt((z_grid-defl_coils).^2+(a+r_grid).^2)).*...
            (((z_grid-defl_coils).^2+r_grid.^2+a^2)./((z_grid-defl_coils).^2+(r_grid-a).^2).*E2_down-E1_down);
        Bz_up=mu0*Itot./(2*pi*sqrt((z_grid+defl_coils).^2+(a+r_grid).^2)).*...
            ((a^2-(z_grid+defl_coils).^2-r_grid.^2)./((z_grid+defl_coils).^2+(r_grid-a).^2).*E2_up+E1_up);
        Bz_down=mu0*Itot./(2*pi*sqrt((z_grid-defl_coils).^2+(a+r_grid).^2)).*...
            ((a^2-(z_grid-defl_coils).^2-r_grid.^2)./((z_grid-defl_coils).^2+(r_grid-a).^2).*E2_down+E1_down);
        Bz=Bz_up+Bz_down;
        Br=Br_up+Br_down;
end
Br(isnan(Br))=0;
% Br=Br/Bz(1,1);
% Bz=Bz/Bz(1,1);

% erase field beyond error - perc
perc=0.05;
% Bz(Bz<Bz(1,1)*(1-perc)|Bz>Bz(1,1)*(1+perc))=0;
r_inds=Bz(1,:)<Bz(1,1)*(1-perc)|Bz(1,:)>Bz(1,1)*(1+perc);
z_inds=Bz(:,1)<Bz(1,1)*(1-perc)|Bz(:,1)>Bz(1,1)*(1+perc);
Bz(z_inds,:)=0;
Bz(:,r_inds)=0;

% visualize remaining field
[zi,ri]=find(Bz);
flat_pts=[zeros(size(ri)),r(ri)',z(zi)';zeros(size(ri)),r(ri)',-z(zi)'];
% plot3(flat_pts(:,1),flat_pts(:,2),flat_pts(:,3),'.')
all_pts=[];
for ang=0:15:360
    all_pts=[all_pts;(rotz(ang)*flat_pts')'];
end
switch plane_flag
    case 'xy'
    case 'yz'
        all_pts=(rotx(90)*all_pts')';
    case 'xz'
        all_pts=(roty(90)*all_pts')';
end
j = boundary(all_pts,1);
bound_pts=all_pts(unique(j(:)),:);
% plot3(all_pts(:,1),all_pts(:,2),all_pts(:,3),'r.')
trisurf(j,all_pts(:,1),all_pts(:,2),all_pts(:,3),'Facecolor','red','FaceAlpha',0.1)
% 
% visualize coils
teta=-pi:(pi/10):pi;
x=a*cos(teta);
y=a*sin(teta);
switch coil_flag
    case 'solenoid'
        switch plane_flag
            case 'xy'
                [x,y,zi] = cylinder(a);
                zi(1, :) = -sol_len/2;
                zi(2, :) = sol_len/2;
                surf(x,y,zi, 'FaceColor', [1,0,0]);
            case 'yz'
                [x,zi,y] = cylinder(a);
                y(1, :) = -sol_len/2;
                y(2, :) = sol_len/2;
                surf(x,y,zi, 'FaceColor', [1,0,0]);
            case 'xz'
                plot3(zeros(1,numel(x)),x,y,'LineWidth',4,'Color','red');
        end
    case 'ring'
        switch plane_flag
            case 'xy'
                plot3(x,y,zeros(1,numel(x)),'LineWidth',4,'Color','red');
            case 'yz'
                plot3(x,zeros(1,numel(x)),y,'LineWidth',4,'Color','red');
            case 'xz'
                plot3(zeros(1,numel(x)),x,y,'LineWidth',4,'Color','red');
        end
    case 'helm'
        switch plane_flag
            case 'xy'
                plot3(x,y,defl_coils*ones(1,numel(x)),'LineWidth',4,'Color','red');
                plot3(x,y,-defl_coils*ones(1,numel(x)),'LineWidth',4,'Color','red');
            case 'yz'
                plot3(x,defl_coils*ones(1,numel(x)),y,'LineWidth',4,'Color','red');
                plot3(x,-defl_coils*ones(1,numel(x)),y,'LineWidth',4,'Color','red');
            case 'xz'
                plot3(defl_coils*ones(1,numel(x)),x,y,'LineWidth',4,'Color','red');
                plot3(-defl_coils*ones(1,numel(x)),x,y,'LineWidth',4,'Color','red');
        end
end

% 2d visualization
% figure
% subplot(1,3,1)
% contour(r,z,Bz)
% title('Bz')
% xlabel('r');ylabel('z')
% axis equal
% 
% subplot(1,3,2)
% quiver(r,z,Br,Bz,1)
% title('B')
% xlabel('r');ylabel('z')
% axis equal
% 
% subplot(1,3,3)
% imagesc(r,z,Bz)
% title('Bz')
% xlabel('r');ylabel('z')
% axis equal
% figure
% quiver(r,z,-Br,Bz,1,'r')
% hold on
% imagesc(r,z,sqrt(Bz.^2+Br.^2))
% title('B')
% xlabel('r');ylabel('z')
% quiver(r,z,Br,Bz,1,'r')
% axis equal
%% magnet2 - horizontal
    %% calculate & visualize magnetic field
a=0.08; % ring radius
sol_len=0.02;

B_desired=1e-2;
tau_desired=5e-4;

Itot=2*B_desired*a/mu0;
I_dens=Itot/sol_len;

wire_rad=0.75e-3/2;
copper_spec_heat= 385;
copper_dens=8960;
    
% I_set=1:5:130;
I_set=10;
pulse_time=5e-3;
ci=0;
for i=1:length(I_set)
    ci=ci+1;
    N(ci)=Itot/I_set(ci);
    wire_length(ci)=2*pi*a*N(ci);
    R_coil(ci)=rho_copper*wire_length(ci)/(pi*(wire_rad)^2);
    
    zeta_p=z_grid+sol_len/2;
    zeta_m=z_grid-sol_len/2;
    alph=@(zet) a./sqrt(zet.^2+(a+r_grid).^2);
    bet=@(zet) zet./sqrt(zet.^2+(a+r_grid).^2);
    k=@(zet) sqrt((zet.^2+(a-r_grid).^2)./(zet.^2+(a+r_grid).^2));
    gamm=(a-r_grid)./(a+r_grid);

    Br=mu0*I_dens/pi*(alph(zeta_p).*...
        gcei(k(zeta_p),ones(size(z_grid)),1,-ones(size(z_grid)))-...
        alph(zeta_m).*gcei(k(zeta_m),ones(size(z_grid)),1,-ones(size(z_grid))));

    Bz=mu0*I_dens/pi*(a./(a+r_grid)).*(bet(zeta_p).*...
        gcei(k(zeta_p),gamm.^2,1,gamm)-...
        bet(zeta_m).*gcei(k(zeta_m),gamm.^2,1,gamm));

    k0=(sol_len/2)/sqrt((sol_len/2)^2+a^2);
    L(ci)=8/3*mu0*(N(ci)/sol_len*a)^2*(sqrt((sol_len/2)^2+a^2)*gcei(k0,1,1,2*k0^2)-a);
    
    R_ext(ci)=L(ci)/tau_desired-R_coil(ci);
    V_out(ci)=I_set(ci)*(R_coil(ci)+R_ext(ci));
    P_out(ci)=V_out(ci)*I_set(ci);
    
    
    wire_mass(ci)=wire_length(ci)*wire_rad^2*pi*copper_dens;
    temp_change(ci)=P_out(ci)*pulse_time/(copper_spec_heat*wire_mass(ci));
end
% plot(I_set,N/max(N))
% hold on 
% plot(I_set,P_out/max(P_out))
% plot(I_set,V_out/max(V_out))
% plot(I_set,R_ext/max(R_ext))
% plot(I_set,L/max(L))
% xlabel('Current [A]')
% legend('N','P_out','V_out','R_ext','L')
% plot(I_set,wire_mass)

% N=Itot/I_set;
% N=30; %number of turns
% I_per_turn=V/(R_coil+R_ext);
% I_per_turn=100;
% Itot=I_per_turn*N;
% R_ext=L/tau_desired-R_coil;
% V_out=I_per_turn*(R_coil+R_ext);
% I_dens=Itot/sol_len;
% P_out=V_out*I_per_turn;

% tau=L/(R_coil+R_ext);
% 0.5*mu0*I/a % ring
%%
coil_flag='solenoid';
plane_flag='xy';
% calculate

switch coil_flag
    case 'solenoid'
        zeta_p=z_grid+sol_len/2;
        zeta_m=z_grid-sol_len/2;
        alph=@(zet) a./sqrt(zet.^2+(a+r_grid).^2);
        bet=@(zet) zet./sqrt(zet.^2+(a+r_grid).^2);
        k=@(zet) sqrt((zet.^2+(a-r_grid).^2)./(zet.^2+(a+r_grid).^2));
        gamm=(a-r_grid)./(a+r_grid);

        Br=mu0*I_dens/pi*(alph(zeta_p).*...
            gcei(k(zeta_p),ones(size(z_grid)),1,-ones(size(z_grid)))-...
            alph(zeta_m).*gcei(k(zeta_m),ones(size(z_grid)),1,-ones(size(z_grid))));
        
        Bz=mu0*I_dens/pi*(a./(a+r_grid)).*(bet(zeta_p).*...
            gcei(k(zeta_p),gamm.^2,1,gamm)-...
            bet(zeta_m).*gcei(k(zeta_m),gamm.^2,1,gamm));
            
        k0=(sol_len/2)/sqrt((sol_len/2)^2+a^2);
        L=8/3*mu0*(N/sol_len*a)^2*(sqrt((sol_len/2)^2+a^2)*gcei(k0,1,1,2*k0^2)-a);
        L_hor=L;
    case 'ring'
        k=4*r_grid*a./(z_grid.^2+(a+r_grid).^2);
        [E1,E2] = ellipke(k);
        Br=mu0*Itot*z_grid./(2*pi*r_grid.*sqrt(z_grid.^2+(a+r_grid).^2)).*...
            ((z_grid.^2+r_grid.^2+a^2)./(z_grid.^2+(r_grid-a).^2).*E2-E1);
        Bz=mu0*Itot./(2*pi*sqrt(z_grid.^2+(a+r_grid).^2)).*...
            ((a^2-z_grid.^2-r_grid.^2)./(z_grid.^2+(r_grid-a).^2).*E2+E1);
    case 'helm'
        k_up=4*r_grid*a./((z_grid+defl_coils).^2+(a+r_grid).^2);
        [E1_up,E2_up] = ellipke(k_up);
        k_down=4*r_grid*a./((z_grid-defl_coils).^2+(a+r_grid).^2);
        [E1_down,E2_down] = ellipke(k_down);
        Br_up=mu0*Itot*(z_grid+defl_coils)./(2*pi*r_grid.*sqrt((z_grid+defl_coils).^2+(a+r_grid).^2)).*...
            (((z_grid+defl_coils).^2+r_grid.^2+a^2)./((z_grid+defl_coils).^2+(r_grid-a).^2).*E2_up-E1_up);
        Br_down=mu0*Itot*(z_grid-defl_coils)./(2*pi*r_grid.*sqrt((z_grid-defl_coils).^2+(a+r_grid).^2)).*...
            (((z_grid-defl_coils).^2+r_grid.^2+a^2)./((z_grid-defl_coils).^2+(r_grid-a).^2).*E2_down-E1_down);
        Bz_up=mu0*Itot./(2*pi*sqrt((z_grid+defl_coils).^2+(a+r_grid).^2)).*...
            ((a^2-(z_grid+defl_coils).^2-r_grid.^2)./((z_grid+defl_coils).^2+(r_grid-a).^2).*E2_up+E1_up);
        Bz_down=mu0*Itot./(2*pi*sqrt((z_grid-defl_coils).^2+(a+r_grid).^2)).*...
            ((a^2-(z_grid-defl_coils).^2-r_grid.^2)./((z_grid-defl_coils).^2+(r_grid-a).^2).*E2_down+E1_down);
        Bz=Bz_up+Bz_down;
        Br=Br_up+Br_down;
end
Br(isnan(Br))=0;
% Br=Br/Bz(1,1);
% Bz=Bz/Bz(1,1);

% erase field beyond error - perc
perc=0.05;
% Bz(Bz<Bz(1,1)*(1-perc)|Bz>Bz(1,1)*(1+perc))=0;
r_inds=Bz(1,:)<Bz(1,1)*(1-perc)|Bz(1,:)>Bz(1,1)*(1+perc);
z_inds=Bz(:,1)<Bz(1,1)*(1-perc)|Bz(:,1)>Bz(1,1)*(1+perc);
Bz(z_inds,:)=0;
Bz(:,r_inds)=0;

% visualize remaining field
[zi,ri]=find(Bz);
flat_pts=[zeros(size(ri)),r(ri)',z(zi)';zeros(size(ri)),r(ri)',-z(zi)'];
% plot3(flat_pts(:,1),flat_pts(:,2),flat_pts(:,3),'.')
all_pts=[];
for ang=0:15:360
    all_pts=[all_pts;(rotz(ang)*flat_pts')'];
end
switch plane_flag
    case 'xy'
    case 'yz'
        all_pts=(rotx(90)*all_pts')';
    case 'xz'
        all_pts=(roty(90)*all_pts')';
end
j = boundary(all_pts,1);
bound_pts=all_pts(unique(j(:)),:);
% plot3(all_pts(:,1),all_pts(:,2),all_pts(:,3),'r.')
trisurf(j,all_pts(:,1),all_pts(:,2),all_pts(:,3),'Facecolor','red','FaceAlpha',0.1)
% 
% visualize coils
teta=-pi:(pi/10):pi;
x=a*cos(teta);
y=a*sin(teta);
switch coil_flag
    case 'solenoid'
        switch plane_flag
            case 'xy'
                [x,y,zi] = cylinder(a);
                zi(1, :) = -sol_len/2;
                zi(2, :) = sol_len/2;
                surf(x,y,zi, 'FaceColor', [1,0,0]);
            case 'yz'
                [x,zi,y] = cylinder(a);
                y(1, :) = -sol_len/2;
                y(2, :) = sol_len/2;
                surf(x,y,zi, 'FaceColor', [1,0,0]);
            case 'xz'
                plot3(zeros(1,numel(x)),x,y,'LineWidth',4,'Color','red');
        end
    case 'ring'
        switch plane_flag
            case 'xy'
                plot3(x,y,zeros(1,numel(x)),'LineWidth',4,'Color','red');
            case 'yz'
                plot3(x,zeros(1,numel(x)),y,'LineWidth',4,'Color','red');
            case 'xz'
                plot3(zeros(1,numel(x)),x,y,'LineWidth',4,'Color','red');
        end
    case 'helm'
        switch plane_flag
            case 'xy'
                plot3(x,y,defl_coils*ones(1,numel(x)),'LineWidth',4,'Color','red');
                plot3(x,y,-defl_coils*ones(1,numel(x)),'LineWidth',4,'Color','red');
            case 'yz'
                plot3(x,defl_coils*ones(1,numel(x)),y,'LineWidth',4,'Color','red');
                plot3(x,-defl_coils*ones(1,numel(x)),y,'LineWidth',4,'Color','red');
            case 'xz'
                plot3(defl_coils*ones(1,numel(x)),x,y,'LineWidth',4,'Color','red');
                plot3(-defl_coils*ones(1,numel(x)),x,y,'LineWidth',4,'Color','red');
        end
end

% 2d visualization
figure
subplot(1,3,1)
contour(r,z,Bz)
title('Bz')
xlabel('r');ylabel('z')
axis equal

subplot(1,3,2)
quiver(r,z,Br,Bz,1)
title('B')
xlabel('r');ylabel('z')
axis equal

subplot(1,3,3)
imagesc(r,z,Bz)
title('Bz')
xlabel('r');ylabel('z')
axis equal
% figure
% quiver(r,z,-Br,Bz,1,'r')
% hold on
% imagesc(r,z,sqrt(Bz.^2+Br.^2))
% title('B')
% xlabel('r');ylabel('z')
% quiver(r,z,Br,Bz,1,'r')
% axis equal
%% close main figure and clear irrelevant variables
% close(figi)
% clear()
%% play sound when finished
toc
load chirp.mat;
sound(y, Fs);
%% clear irrelevant variables
clear('y','Fs','save_flag','save_dir','realFrameRate','plot_flag',...
    'outputVideo','frame_to_ms','dataPath','num_flaps','n_frames',...
    'figs_bod','figs_pitch','figs_pitch_map1','figs_pitch_map2','figs_wings',...
    'cbh','num_of_clusters','num_of_pitch_vecs')