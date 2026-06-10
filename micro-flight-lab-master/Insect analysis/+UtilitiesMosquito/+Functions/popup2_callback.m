function popup2_callback(h,~,ax_ci,wing1_edges_fat,wing2_edges_fat,ims_rem_orig,pp1)
    all_poss_matches=[1,2,3;1,2,6;1,5,3;1,5,6];
    all_poss_matches=[all_poss_matches;mod(all_poss_matches+[2,2,2],6)+1];
    
    inds_1=all_poss_matches(pp1.Value,:);
    inds_2=all_poss_matches(h.Value,:);
    
    for cam_ind=1:3
        full_im_with_edges=2^16*wing1_edges_fat{inds_1(cam_ind)}+...
            ims_rem_orig{cam_ind}-2^16*wing2_edges_fat{inds_2(cam_ind)};
        [y,x]=find(full_im_with_edges);
        ax_ci{cam_ind}=subplot(1,3,cam_ind);
        imshow(full_im_with_edges,[],'Parent',ax_ci{cam_ind});
        xlim([min(x),max(x)])
        ylim([min(y),max(y)])
        title([num2str(cam_ind),' white wing is right/1'])
    end
end