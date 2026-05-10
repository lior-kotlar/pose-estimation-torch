function preds3d_smooth = smooth_3d_points(pts_to_smooth, ThresholdFactor, pvalue)
num_joints=size(pts_to_smooth,1);
dt=1/16000;
n_frames=size(pts_to_smooth, 2);
preds3d_smooth=nan(num_joints,n_frames,3);
x_s=0:dt:(dt*(n_frames-1));
outlier_window_size=21;
p=pvalue;
% p=0.99995;
% p=0.999999;
for node_ind=1:num_joints
    for dim_ind=1:3
        tmp_rm=squeeze(pts_to_smooth(node_ind,:,dim_ind));
        [~,rm_inds]=rmoutliers(tmp_rm,'movmedian',outlier_window_size,'ThresholdFactor',ThresholdFactor);
        tmp_rm(rm_inds)=nan;
        % remove outliners from best_err_pts_all
        pts_to_smooth(node_ind, :, dim_ind) = tmp_rm;
        preds3d_rm(node_ind,dim_ind,:)=tmp_rm;
        [xdata, ~, stdData ] = curvefit.normalize(x_s);
        pps(node_ind,dim_ind) = csaps(xdata,tmp_rm,p); 
        preds3d_smooth(node_ind,:,dim_ind)=fnval(pps(node_ind,dim_ind),xdata);
    end
end
end