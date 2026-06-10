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
dataPath='E:\Igal\2018_09_05_igal_cutleg\with_legs_A\mov2';
pics_for_edges_dir='C:\google drive\Micro Flight Group\Code\MatlabCode\Noam\pics_for_edges';
%% load easywand data and create the all_cameras_class
easyWandData=load([dataPath,'\easyWandData.mat']);
allCams=all_cameras_class(easyWandData.easyWandData);
%% load sparse_movie files and load backgrounds to cameras
sparseLRFilenames = cell(1,3) ;
sparseLRFilenames{1} = [dataPath '\mov2_cam2_sparse_array_LR.mat'] ;
sparseLRFilenames{2} = [dataPath '\mov2_cam3_sparse_array_LR.mat'] ;
sparseLRFilenames{3} = [dataPath '\mov2_cam4_sparse_array_LR.mat'] ;
sparseorigFilenames = cell(1,3) ;
sparseorigFilenames{1} = [dataPath '\mov2_cam2_sparse_array.mat'] ;
sparseorigFilenames{2} = [dataPath '\mov2_cam3_sparse_array.mat'] ;
sparseorigFilenames{3} = [dataPath '\mov2_cam4_sparse_array.mat'] ;

sparse_movies_LR=cell(3,1);
sparse_movies=cell(3,1);
for cam_ind=1:numel(sparseLRFilenames)
    loady=load(sparseLRFilenames{cam_ind}); %loads sparse_array
    sparse_movies_LR{cam_ind}=loady.sparse_array;
    loady=load(sparseorigFilenames{cam_ind}); %loads sparse_array
    sparse_movies{cam_ind}=loady.sparse_array;
    % background is saved in first element of sparse_array
    allCams.cams_array(cam_ind).load_background(loady.sparse_array{1});
end
%% set parameters and initialize variables
realFrameRate=20000; % !!! fix to get from file/xml

frames_per_flap=34; % number of frames per flap
num_flaps=1; % number of flaps
start_fr=2; % starting frame number (movie frames start from 2 in sparse_array)
orig_offset=525;

% n_frames=frames_per_flap*num_flaps; % number of frames to loop on
n_frames=length(sparse_movies_LR{1})-1;

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
    save_dir='figs\lr1\';
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
        allCams.cams_array(cam_ind).load_image(mosquito_image_class(...
            bwareaopen(imopen(full(sparse_movies_LR{cam_ind}{fr}),strel('disk',0)),40).*...
            full(sparse_movies_LR{cam_ind}{fr})));
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
    %% calculate body angles
    mosquito_all_frames(c).get_body_angs;
    body_angs(c,:)=mosquito_all_frames(c).body_angs;
    %% plot hull
    if plot_flag
        arrow_scale=0.005;
        hold on
        color_cell=mat2cell(lines(5),[1,1,1,1,1],3);
        cellfun(@(x,y) pcshow(x,y,'MarkerSize',12), mosquito_all_frames(c).all_clusters(1:4),color_cell(1:4))
        
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
    %% save figures and images
    if save_flag
        close(outputVideo);
        savefig(figs_bod,[save_dir,'body_angs'],'compact')
        saveas(figs_bod,[save_dir,'body_angs.png'])
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