%% Description:
% Loads a sparse_movie and calculate body & wing angles usingexisting
% images with edges marked

% manually analyzed for lora_data
% max_cord_length=6.7e-4
% span_length=2.6e-3
% air_dyn_visc=15.4e-6;
% Re = span_length*mean(abs(diff(wing_stroke))/frame_to_ms/1000)*max_cord_length/air_dyn_visc=94
% freq = 588Hz
%% set directories and path
tic
%home
% addpath(genpath('C:\Users\Noamler\Documents\git reps\micro_flight_lab\Insect analysis'));
% dataPath='G:\My Drive\Micro Flight Group\Videos\2018_08_29_igal_cutleg';
% pics_for_edges_dir='G:\My Drive\Micro Flight Group\Code\MatlabCode\Noam\pics_for_edges';
%lab
addpath(genpath('C:\git reps\micro_flight_lab\Insect analysis'));
dataPath='C:\google drive\Micro Flight Group\Videos\2018_08_29_igal_cutleg';
pics_for_edges_dir='C:\google drive\Micro Flight Group\Code\MatlabCode\Noam\pics_for_edges';
%% load easywand data and create the all_cameras_class
easyWandData=load([dataPath,'\29_08_2018_easyWandData.mat']);
allCams=all_cameras_class(easyWandData.easyWandData);
%% load sparse_movie files and load backgrounds to cameras
sparseFilenames = cell(1,3) ;
sparseFilenames{1} = [dataPath '\mov2_cam2_sparse_array.mat'] ;
sparseFilenames{2} = [dataPath '\mov2_cam3_sparse_array.mat'] ;
sparseFilenames{3} = [dataPath '\mov2_cam4_sparse_array.mat'] ;
sparse_movies=cell(3,1);
for cam_ind=1:numel(sparseFilenames)
    loady=load(sparseFilenames{cam_ind}); %loads sparse_array
    sparse_movies{cam_ind}=loady.sparse_array;
    % background is saved in first element of sparse_array
    allCams.cams_array(cam_ind).load_background(loady.sparse_array{1});
end
%% set parameters and initialize variables

realFrameRate=20000; % !!! fix to get from file/xml

frames_per_flap=34; % number of frames per flap
num_flaps=1; % number of flaps
start_fr=1081; % starting frame number (movie frames start from 2 in sparse_array)
n_frames=frames_per_flap*num_flaps; % number of frames to loop on
frame_to_ms=1000/realFrameRate; % conversion from frame number to time[ms]
cam_inds_rec=2:3; % camera indices to use in recunstrucion of edges
num_of_pitch_vecs=6; % number of pitch angles to measure on each wing
plot_flag=true; 
save_flag=true;
c=0;
% parameters for grid reconstruction
voxelSize = 50e-6 ; % size of grid voxel
volLength= 14e-3 ; % size of the square sub-vol cube to reconstruct (meters)
offset_index_size=90; % size of search area when original seed is not a true voxel

how_much_tail=0.67;
wing_flatness_thresh=3.3;
num_of_clusters=5; % number of clusters to use in kmeans
cents_in=[];
cluster_params=cluster_and_wings_params_class(how_much_tail,...
                wing_flatness_thresh,num_of_clusters,cents_in);
% initialize variables
mosquito_all_frames(n_frames,1)=mosquito_frame_class; % holds all data for each frame (clusters,angles,...)
wing_tip=zeros(n_frames,2,3);
pitchmap1=zeros(num_of_pitch_vecs*n_frames,3);
pitchmap2=zeros(num_of_pitch_vecs*n_frames,3);
body_angs=zeros(n_frames,3);
wing_angs=zeros(2,3,n_frames);
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
    outputVideo = VideoWriter([save_dir,'mov.avi']);
    outputVideo.FrameRate = 10;
    outputVideo.Quality = 100;
    open(outputVideo);
end
%% clear irrelevant variables
clear('loady','sparseFilenames','easyWandData','frames_per_flap',...
    'how_much_tail','wing_flatness_thresh','num_of_clusters','cents_in')
%% loop on frames
for fr=start_fr:(start_fr+n_frames-1)
    c=c+1;
    disp(['frame ',num2str(c),' from ',num2str(n_frames)])
    %% create hull from original images
    for cam_ind=1:3
        allCams.cams_array(cam_ind).load_image(mosquito_image_class(sparse_movies{cam_ind}{fr}));
    end
    allCams.load_ims_for_reconstruction('Image'); % updates images and seed for reconstruction

    [hull_params]=hull_params_class(allCams.currCM3D,voxelSize,volLength,offset_index_size);
    [ hull_inds,~] = hull_reconstruction_on_grid( allCams,hull_params );
    % translate to easywand space
    hull=[hull_params.real_coord(1,hull_inds(:,1))',hull_params.real_coord(2,hull_inds(:,2))',...
        hull_params.real_coord(3,hull_inds(:,3))'];
    %% cluster and clean original hull
    % cluster original hull
    mosquito_all_frames(c).all_clusters=clusters_simple(hull,cluster_params);
    % cast body parts and binarize seperatly to remove hairs etc.
    for cam_ind=1:3
        body_casted=bwareafilt(imclose(threeDtoImage(allCams.all_DLT_coefs(:,cam_ind),...
            cell2mat(mosquito_all_frames(c).all_clusters(3:5))),strel('disk',3)),1);
        thinner_full_mask=(mat2gray(full(allCams.cams_array(cam_ind).curr_im.ImageWOBG).*body_casted)>0.4)+...
            ((full(allCams.cams_array(cam_ind).curr_im.Image)>0).*~body_casted);
        allCams.cams_array(cam_ind).load_image(mosquito_image_class(sparse(thinner_full_mask.*...
            full(allCams.cams_array(cam_ind).curr_im.Image))));
    end
    allCams.load_ims_for_reconstruction('Image'); % updates images an seed for reconstruction
    % reconstruct from clean images
    [hull_params]=hull_params_class(allCams.currCM3D,voxelSize,volLength,offset_index_size);
    [ hull_inds,~] = hull_reconstruction_on_grid( allCams,hull_params );
    hull=[hull_params.real_coord(1,hull_inds(:,1))',hull_params.real_coord(2,hull_inds(:,2))',...
        hull_params.real_coord(3,hull_inds(:,3))'];
    % cluster clean hull and fix tail/wings
    [mosquito_all_frames(c).all_clusters,mosquito_all_frames(c).evecs]=...
        clusters_and_wings(hull,cluster_params);
    %% rotate to lab frame
    mosquito_all_frames(c).all_clusters=cellfun(@(x) x*allCams.Rotation_Matrix',mosquito_all_frames(c).all_clusters,'UniformOutput',0);
    mosquito_all_frames(c).evecs=cellfun(@(x) allCams.Rotation_Matrix*x,mosquito_all_frames(c).evecs,'UniformOutput',0);
    mosquito_all_frames(c).cents=cell2mat(cellfun(@(x) mean(x), mosquito_all_frames(c).all_clusters,'UniformOutput',0));
    %% save coordinates of wing tips
    for wing_ind=1:2
        wing_dist=( mosquito_all_frames(c).all_clusters{wing_ind}-mosquito_all_frames(c).cents(5,:));
        [~,I]=max(sum(wing_dist.^2,2));
        wing_tip(c,wing_ind,:)= mosquito_all_frames(c).all_clusters{wing_ind}(I,:);
        mosquito_all_frames(c).wing_tips(wing_ind,:)=wing_tip(c,wing_ind,:);
    end
    %% calculate body angles
    mosquito_all_frames(c).get_body_angs;
    body_angs(c,:)=mosquito_all_frames(c).body_angs;
    %% generate wing edges hulls
    for cam_ind=1:3
        % save images for edge marking: red - 1 front, green - 1 back, blue- 2 front, yellow - 2 back
%         imwrite(ims_wobg{1},['pics_for_edges\',num2str(fr),'_',num2str(cam),'.png'])
        im_edges_load=imread([pics_for_edges_dir,'\',num2str(fr),'_',num2str(cam_ind),'.png']);
        
        allCams.cams_array(cam_ind).curr_im.wing_r_front_edge=...
            imdilate((im_edges_load(:,:,1)==255)&(im_edges_load(:,:,2)==0)&(im_edges_load(:,:,3)==0),...
            strel('disk',1))&(full(allCams.cams_array(cam_ind).curr_im.Image)>0);
        allCams.cams_array(cam_ind).curr_im.wing_r_back_edge=...
            imdilate((im_edges_load(:,:,1)==0)&(im_edges_load(:,:,2)==255)&(im_edges_load(:,:,3)==0),...
            strel('disk',1))&(full(allCams.cams_array(cam_ind).curr_im.Image)>0);
        allCams.cams_array(cam_ind).curr_im.wing_l_front_edge=...
            imdilate((im_edges_load(:,:,1)==0)&(im_edges_load(:,:,2)==0)&(im_edges_load(:,:,3)==255),...
            strel('disk',1))&(full(allCams.cams_array(cam_ind).curr_im.Image)>0);
        allCams.cams_array(cam_ind).curr_im.wing_l_back_edge=...
            imdilate((im_edges_load(:,:,1)==255)&(im_edges_load(:,:,2)==255)&(im_edges_load(:,:,3)==0),...
            strel('disk',1))&(full(allCams.cams_array(cam_ind).curr_im.Image)>0);
    end
    
    allCams.load_ims_for_reconstruction('wing_r_front_edge'); % updates images an seed for reconstruction
    mosquito_all_frames(c).wing_r_front_hull=hull_reconstruction_from_ims(allCams,cam_inds_rec)*allCams.Rotation_Matrix';
    allCams.load_ims_for_reconstruction('wing_r_back_edge'); % updates images an seed for reconstruction
    mosquito_all_frames(c).wing_r_back_hull=hull_reconstruction_from_ims(allCams,cam_inds_rec)*allCams.Rotation_Matrix';
    allCams.load_ims_for_reconstruction('wing_l_front_edge'); % updates images an seed for reconstruction
    mosquito_all_frames(c).wing_l_front_hull=hull_reconstruction_from_ims(allCams,cam_inds_rec)*allCams.Rotation_Matrix';
    allCams.load_ims_for_reconstruction('wing_l_back_edge'); % updates images an seed for reconstruction
    mosquito_all_frames(c).wing_l_back_hull=hull_reconstruction_from_ims(allCams,cam_inds_rec)*allCams.Rotation_Matrix';
    %% calculate wing angles
    mosquito_all_frames(c).get_wing_angs_edges(num_of_pitch_vecs);
    wing_angs(:,:,c)=mosquito_all_frames(c).wing_angs;
    
    pitches1=mosquito_all_frames(c).pitches{1,1};
    front1_points3d=pitches1(:,1:3)+pitches1(:,4:6);
    dists_from_tip1=vecnorm(front1_points3d-squeeze(wing_tip(c,1,:))',2,2);
    pitches2=mosquito_all_frames(c).pitches{2,1};
    front2_points3d=pitches2(:,1:3)+pitches2(:,4:6);
    dists_from_tip2=vecnorm(front2_points3d-squeeze(wing_tip(c,2,:))',2,2);
    
    pitchmap1((num_of_pitch_vecs*(c-1)+1):(num_of_pitch_vecs*(c-1)+num_of_pitch_vecs),:)=...
        [ones(num_of_pitch_vecs,1)*c,dists_from_tip1*1000,mosquito_all_frames(c).pitches{1,2}];    
    pitchmap2((num_of_pitch_vecs*(c-1)+1):(num_of_pitch_vecs*(c-1)+num_of_pitch_vecs),:)=...
        [ones(num_of_pitch_vecs,1)*c,dists_from_tip2*1000,mosquito_all_frames(c).pitches{2,2}];
    %% plot hull
    if plot_flag
        arrow_scale=0.005;
        hold on
        color_cell=mat2cell(lines(5),[1,1,1,1,1],3);
        cellfun(@(x,y) pcshow(x,y,'MarkerSize',12), mosquito_all_frames(c).all_clusters(1:4),color_cell(1:4))
        pcshow([mosquito_all_frames(c).wing_r_front_hull;mosquito_all_frames(c).wing_r_back_hull],...
            'r','MarkerSize',12)
        pcshow([mosquito_all_frames(c).wing_l_front_hull;mosquito_all_frames(c).wing_l_back_hull],...
            'b','MarkerSize',12)
        all_pitches=[pitches1;pitches2];

        for pitch_vec_ind=1:size(all_pitches,1)
            quiver3(all_pitches(pitch_vec_ind,1),all_pitches(pitch_vec_ind,2),all_pitches(pitch_vec_ind,3),...
                all_pitches(pitch_vec_ind,4),all_pitches(pitch_vec_ind,5),all_pitches(pitch_vec_ind,6),'Color','k')
        end
        
        if c<8
            edge_frames=1:c;
        else
            edge_frames=(c-6):c;
        end
        
        plot3(wing_tip(edge_frames,1,1),wing_tip(edge_frames,1,2),wing_tip(edge_frames,1,3),...
            'Marker','.','MarkerSize',12,'MarkerEdgeColor','k')
        plot3(wing_tip(edge_frames,2,1),wing_tip(edge_frames,2,2),wing_tip(edge_frames,2,3),...
            'Marker','.','MarkerSize',12,'MarkerEdgeColor','k')
        
        plane_size=0.0025;
        fsurf(@(s,t) mosquito_all_frames(c).cents(5,1)+mosquito_all_frames(c).stroke_plane(1,1)*s+mosquito_all_frames(c).stroke_plane(1,2)*t,...
            @(s,t) mosquito_all_frames(c).cents(5,2)+mosquito_all_frames(c).stroke_plane(2,1)*s+mosquito_all_frames(c).stroke_plane(2,2)*t,...
            @(s,t) mosquito_all_frames(c).cents(5,3)+mosquito_all_frames(c).stroke_plane(3,1)*s+mosquito_all_frames(c).stroke_plane(3,2)*t,...
        [-plane_size,plane_size,-plane_size,plane_size],'EdgeColor','none','FaceAlpha',0.3)
        
        plot_camera_surfaces(allCams,mosquito_all_frames(c).cents(5,:),0.5e-2)

        axisLimitMult=1.5;
        if c==1
            xlimits=axisLimitMult*(axi.XLim-mean(axi.XLim))+mean(axi.XLim);
            ylimits=axisLimitMult*(axi.YLim-mean(axi.YLim))+mean(axi.YLim);
            zlimits=axisLimitMult*(axi.ZLim-mean(axi.ZLim))+mean(axi.ZLim);
        end
        xlim(xlimits)
        ylim(ylimits)
        zlim(zlimits)
        grid off
        view(c/n_frames*45*num_flaps,36)
        title(['frame number: ',num2str(fr)])

        if save_flag
            savefig(figi,[save_dir,num2str(fr)],'compact')
            writeVideo(outputVideo, getframe(figi));
            cla(axi)
        end
    end
end
%% close main figure and clear irrelevant variables
close(figi)
clear('axisLimitMult','xlimits','ylimits','zlimits',...
    'plane_size','edge_frames','c','fr','arrow_scale',...
    'color_cell','thinner_full_mask','cam_inds_rec','im_edges_load',...
    'allCams','axi','figi','voxelSize','cluster_params',...
    'volLength','offset_index_size','start_fr','pitch_vec_ind','pics_for_edges_dir',...
    'sparse_movies','wing_ind','I','hull_params','cam_ind','hull','hull_inds',...
    'body_casted','kmeans_cents_prev','pitches1','pitches2','all_pitches',...
    'dists_from_tip1','dists_from_tip2','front1_points3d','front2_points3d','wing_dist')
%% plot and save data 
if plot_flag
    %% body angles
    figs_bod=figure;
    plot((1:n_frames),body_angs,'Marker','o','MarkerFaceColor','auto')
    legend('pitch','yaw','roll')
    xlabel('Frame No.')
    ylabel('Angle[°]')
    title('Body Angles')
    %% wing angles
    figs_wings=figure;
    hold on
    plot((1:n_frames),squeeze(wing_angs(1,:,:)),'Marker','o','MarkerFaceColor','auto')
    plot((1:n_frames),squeeze(wing_angs(2,:,:)),'Marker','o','MarkerFaceColor','auto')
    legend('stroke1_ang','elav1_ang','pitch1_ang','stroke2_ang','elav2_ang','pitch2_ang')
    xlabel('Frame No.')
    ylabel('Angle[°]')
    title('Wing Angles')
    %% pitch angles
    figs_pitch=figure;
    hold on
    pitchmap1= sortrows(pitchmap1, [1,-3]) ;
    pitch1_fix=reshape(pitchmap1(:,3),num_of_pitch_vecs,[]);
    pitchmap2= sortrows(pitchmap2, [1,-3]) ;
    pitch2_fix=reshape(pitchmap2(:,3),num_of_pitch_vecs,[]);
    plot((1:n_frames),pitch1_fix,'Marker','o','MarkerFaceColor','auto')
    plot((1:n_frames),-pitch2_fix,'Marker','o','MarkerFaceColor','auto')% minus for visualization
    xlabel('Frame No.')
    ylabel('Angle[°]')
    title('Pitch Angles')
    %% wing angle dependencies
    figure
    hold on
    plot(squeeze(wing_angs(1,1,:)),squeeze(wing_angs(1,2,:)),'Marker','o','MarkerFaceColor','auto')
    plot(squeeze(wing_angs(2,1,:)),squeeze(wing_angs(2,2,:)),'Marker','o','MarkerFaceColor','auto')
    axis equal
    figure
    hold on
    plot(squeeze(wing_angs(1,1,:)),pitch1_fix,'Marker','o','MarkerFaceColor','auto')
    axis equal
    %% pitchmaps
    figs_pitch_map1=figure;
    hold on
    pitchmap1(any(isnan(pitchmap1), 2), :) = [];
    trisurf(delaunay(pitchmap1(:,1),pitchmap1(:,2)),pitchmap1(:,1),pitchmap1(:,2),pitchmap1(:,3));
    xlabel('Frame No.')
    ylabel('DistanceFrom tip')
    zlabel('Angle[°]')
    title('Pitch Map wing1')
    axis vis3d
    lighting phong
    shading interp
    cbh=colorbar('EastOutside');
    ylabel(cbh, 'Pitch Angle[°]')
    colormap hot

    figs_pitch_map2=figure;
    hold on
    pitchmap2(any(isnan(pitchmap2), 2), :) = [];
    trisurf(delaunay(pitchmap2(:,1),pitchmap2(:,2)),pitchmap2(:,1),pitchmap2(:,2),pitchmap2(:,3));
    xlabel('Frame No.')
    ylabel('DistanceFrom tip')
    zlabel('Angle[°]')
    title('Pitch Map wing2')
    axis vis3d
    lighting phong
    shading interp
    cbh=colorbar('EastOutside');
    ylabel(cbh, 'Pitch Angle[°]')
    colormap hot
    % both wings
    figure
    hold on
    trisurf(delaunay(pitchmap1(:,1),-pitchmap1(:,2)+2.6),pitchmap1(:,1),-pitchmap1(:,2)+2.6,pitchmap1(:,3));
    trisurf(delaunay(pitchmap2(:,1),pitchmap2(:,2)-2.6),pitchmap2(:,1),pitchmap2(:,2)-2.6,pitchmap2(:,3));
    xlabel('Frame No.')
    ylabel('DistanceFrom base')
    zlabel('Angle[°]')
    title('Pitch Map wings')
    axis vis3d
    lighting phong
    shading interp
    cbh=colorbar('EastOutside');
    ylabel(cbh, 'Pitch Angle[°]')
    colormap hot
    %% save figures and images
    if save_flag
        close(outputVideo);
        savefig(figs_bod,[save_dir,'body_angs'],'compact')
        saveas(figs_bod,[save_dir,'body_angs.png'])
        savefig(figs_wings,[save_dir,'wing_angs'],'compact')
        saveas(figs_wings,[save_dir,'wing_angs.png'])
        savefig(figs_pitch,[save_dir,'pitch_angs'],'compact')
        saveas(figs_pitch,[save_dir,'pitch_angs.png'])
        savefig(figs_pitch_map1,[save_dir,'pitch_map1'],'compact')
        saveas(figs_pitch_map1,[save_dir,'pitch_map1.png'])
        savefig(figs_pitch_map2,[save_dir,'pitch_map2'],'compact')
        saveas(figs_pitch_map2,[save_dir,'pitch_map2.png'])
    end
end
%% play sound when finished
toc
load chirp.mat;
sound(y, Fs);
%% clear irrelevant variables
clear('y','Fs','save_flag','save_dir','realFrameRate','plot_flag',...
    'outputVideo','frame_to_ms','dataPath','num_flaps','n_frames',...
    'figs_bod','figs_pitch','figs_pitch_map1','figs_pitch_map2','figs_wings',...
    'cbh','num_of_clusters','num_of_pitch_vecs')