function [errs, best_err_pts_all ,all_pts3d] = get_3d_pts_rays_intersects(predictions, easyWandData, cropzone, cam_inds)
    %% set variabes
    num_joints=size(predictions,3);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
    num_cams=length(allCams.cams_array);
    centers=allCams.all_centers_cam';
    couples=nchoosek(1:num_cams,2);
    num_couples=size(couples,1);
    n_frames=size(predictions,1);
    all_pts3d=nan(num_joints,n_frames,num_couples,3);
    %% get body points in 3d from all couples sigma_2_5_3_cams
    best_errors = nan(num_joints, n_frames);
    for frame_ind=1:n_frames
        for node_ind=1:num_joints
            frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
           
            x=double(cropzone(2,cam_inds,frame_ind)) + double(squeeze(predictions(frame_ind,:,node_ind, 1)));
            y=double(cropzone(1,cam_inds,frame_ind)) + double(squeeze(predictions(frame_ind,:,node_ind, 2)));
    
            PB=nan(length(cam_inds),4);
            for cam=1:num_cams
                PB(cam,:)=allCams.cams_array(cam_inds(cam)).invDLT * [x(cam); (801-y(cam)); 1];
            end
            
            % calculate all couples
            for couple_ind=1:size(couples,1)
                couple = couples(couple_ind,:);
                normalized_PB = PB(couple,1:3)./PB(couple,4);
                PA = centers(cam_inds(couple),:);
                [pt3d_candidates(couple_ind,:),errs(node_ind,frame_ind,couple_ind,:)]=...
                    HullReconstruction.Functions.lineIntersect3D(PA,normalized_PB);
            end
            all_pts3d(node_ind,frame_ind,:,:)=pt3d_candidates*allCams.Rotation_Matrix';
            
            [best_err,best_err_ind]=min(squeeze(errs(node_ind,frame_ind,:,1)));
            best_errors(node_ind, frame_ind) = best_err;
            best_err_pt=pt3d_candidates(best_err_ind,:)*allCams.Rotation_Matrix';
            best_err_pts_all(node_ind,frame_ind,:)=best_err_pt;
        end
    end
    errs = errs(:,:,:,1);
end