function [all_errors_ensamble, all_pts_ensemble_3d] = get_all_ensemble_pts(num_joints, n_frames ,num_cams ,easyWandData, cropzone, box)
        num_wings_pts= 14;
        pnt_per_wing = num_wings_pts/2;
        left_inds = 1:num_wings_pts/2; right_inds = (num_wings_pts/2+1:num_wings_pts); 
        head_tail_inds = (num_wings_pts + 1:num_wings_pts + 2);
        couples=nchoosek(1:num_cams,2);
        num_couples=size(couples,1);
        ensemble_preds_paths = [];
        for i=0:5
            path = strcat("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 2 good cameras\sigma 3\train_on_2_good_cameras_seed_", string(i) ,"_sigma_3\predict_over_movie.h5");
            ensemble_preds_paths = [ensemble_preds_paths, path];
        end
        ensemble_preds_paths;
        num_of_models = size(ensemble_preds_paths,2);
        all_pts_ensemble_3d = nan(num_joints,n_frames,num_couples*num_of_models, 3);
        all_errors_ensamble = nan(num_joints,n_frames,num_couples*num_of_models);
        for i=1:num_of_models
            preds_path = ensemble_preds_paths(i);
            % arrange predictions
            preds_i = h5read(preds_path,'/positions_pred');
            preds_i = single(preds_i) + 1;
            predictions_i = rearange_predictions(preds_i, num_cams);
            head_tail_predictions_i = predictions_i(:,:,head_tail_inds,:);
            predictions_i = predictions_i(:,:,1:num_wings_pts,:);
            % fix predictions per camera 
            [predictions_i, ~] = fix_wings_per_camera(predictions_i, box);
            % fix wing 1 and wing 2 
            [predictions_i, ~] = fix_wings_3d(predictions_i, easyWandData, cropzone, box, false);
            % get 3d pts from 4 2d cameras 
            predictions_i(:,:,head_tail_inds,:) = head_tail_predictions_i;
            [all_errors_i, best_err_pts_all_i, all_pts3d_i] = get_3d_pts_rays_intersects(predictions_i, easyWandData, cropzone, 1:num_cams);
            all_errors_i = all_errors_i(:,:,:,1);
            pairs_indexes = num_couples*(i-1) + (1:num_couples);
            all_pts_ensemble_3d(:,:, pairs_indexes,:) = all_pts3d_i;
            all_errors_ensamble(:,:, pairs_indexes) = all_errors_i;
        end
    end