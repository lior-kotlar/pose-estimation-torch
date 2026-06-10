function hull_pts=hull_reconstruction_from_ims(all_cams,cam_inds_rec)
% Description:
% Constructor 
% 
% Required input:
% all_cams - current frame grayscale image in sparse format
% cam_inds_rec
%
% Output:
% hull_pts- image_class
    h=all_cams.size_image(1);
    w=all_cams.size_image(2);
    c=0;
    row=cell(length(cam_inds_rec),1);
    col=cell(length(cam_inds_rec),1);
    for cam_ind=cam_inds_rec
        c=c+1;
        [row{c},col{c}]=find(all_cams.cams_array(cam_ind).bin_image_for_recon.Image);
    end
    
    fits=[];
    dist_thresh=0.8;
    %%
    if length(cam_inds_rec)==2
        lines23all=all_cams.Fundamental_matrices(:,:,3)*[col{1},h+1-row{1},ones(length(row{1}),1)]';
        lines23all(3,:)=lines23all(3,:)+(h+1)*lines23all(2,:); % transform back to image coordinates
        lines23all(2,:)=-lines23all(2,:);
        lines23all=-lines23all./lines23all(2,:);
        for i=1:length(col{1})
            dists12=dist_ln_pt(lines23all(:,i)',[col{2}';row{2}';ones(1,length(row{2}))]);
            [B,sort_inds]=sort(dists12);
            might_fit_inds2=sort_inds(B<dist_thresh);

            might_fit_col2=col{2}(might_fit_inds2);
            might_fit_row2=row{2}(might_fit_inds2);

            if isempty(might_fit_inds2)
                continue
            end
            new_fits=zeros(length(might_fit_inds2),2);
            new_fits(:,1)=sub2ind([h,w],row{1}(i),col{1}(i))*ones(size(might_fit_inds2,1),1);
            new_fits(:,2)=sub2ind([h,w], might_fit_row2, might_fit_col2);
            fits=[fits;new_fits];
        end
    else
        lines12all=all_cams.Fundamental_matrices(:,:,1)*[col{1},h+1-row{1},ones(length(row{1}),1)]';
        lines12all(3,:)=lines12all(3,:)+(h+1)*lines12all(2,:); % transform back to image coordinates
        lines12all(2,:)=-lines12all(2,:);
        lines12all=-lines12all./lines12all(2,:);

        lines13all=all_cams.Fundamental_matrices(:,:,2)*[col{1},h+1-row{1},ones(length(row{1}),1)]';
        lines13all(3,:)=lines13all(3,:)+(h+1)*lines13all(2,:); % transform back to image coordinates
        lines13all(2,:)=-lines13all(2,:);
        lines13all=-lines13all./lines13all(2,:);

        lines23all=all_cams.Fundamental_matrices(:,:,3)*[col{2},h+1-row{2},ones(length(row{2}),1)]';
        lines23all(3,:)=lines23all(3,:)+(h+1)*lines23all(2,:); % transform back to image coordinates
        lines23all(2,:)=-lines23all(2,:);
        lines23all=-lines23all./lines23all(2,:);

        lines32all=all_cams.Fundamental_matrices(:,:,3)'*[col{3},h+1-row{3},ones(length(row{3}),1)]';
        lines32all(3,:)=lines32all(3,:)+(h+1)*lines32all(2,:); % transform back to image coordinates
        lines32all(2,:)=-lines32all(2,:);
        lines32all=-lines32all./lines32all(2,:);

        lines31all=all_cams.Fundamental_matrices(:,:,2)'*[col{3},h+1-row{3},ones(length(row{3}),1)]';
        lines31all(3,:)=lines31all(3,:)+(h+1)*lines31all(2,:); % transform back to image coordinates
        lines31all(2,:)=-lines31all(2,:);
        lines31all=-lines31all./lines31all(2,:);
        
        for i=1:length(col{1})
            dists12=dist_ln_pt(lines12all(:,i)',[col{2}';row{2}';ones(1,length(row{2}))]);
            [B,sort_inds]=sort(dists12);
            might_fit_inds2=sort_inds(B<dist_thresh);
            dists13=dist_ln_pt(lines13all(:,i)',[col{3}';row{3}';ones(1,length(row{3}))]);
            [B,sort_inds]=sort(dists13);
            might_fit_inds3=sort_inds(B<dist_thresh);

            might_fit_col2=col{2}(might_fit_inds2);
            might_fit_row2=row{2}(might_fit_inds2);
            might_fit_col3=col{3}(might_fit_inds3);
            might_fit_row3=row{3}(might_fit_inds3);

            lines23=all_cams.Fundamental_matrices(:,:,3)*[might_fit_col2,...
                h+1-might_fit_row2,ones(length(might_fit_inds2),1)]';
            lines23(3,:)=lines23(3,:)+(h+1)*lines23(2,:); % transform back to image coordinates
            lines23(2,:)=-lines23(2,:);
            lines23=-lines23./lines23(2,:);

            lines32=all_cams.Fundamental_matrices(:,:,3)'*[might_fit_col3,...
                h+1-might_fit_row3,ones(length(might_fit_inds3),1)]';
            lines32(3,:)=lines32(3,:)+(h+1)*lines32(2,:); % transform back to image coordinates
            lines32(2,:)=-lines32(2,:);
            lines32=-lines32./lines32(2,:);

            dists23=dist_ln_pt(lines23',[might_fit_col3';might_fit_row3';...
                ones(1,length(might_fit_row3))]);
            [B,sort_inds]=sort(dists23(:));
            [r,c] = ind2sub(size(dists23),sort_inds(B<dist_thresh));

            dists32=dist_ln_pt(lines32',[might_fit_col2';might_fit_row2';...
                ones(1,length(might_fit_row2))]);
            [B,sort_inds]=sort(dists32(:));
            [r2,c2] = ind2sub(size(dists32),sort_inds(B<dist_thresh));

            ddd=[r,c;c2,r2];
            ddd_un=unique(ddd,'rows');

            if any(size(ddd_un)==0)
%                 pt_list=[];
                continue
            end
            new_fits=zeros(size(ddd_un,1),3);
            new_fits(:,1)=sub2ind([h,w],row{1}(i),col{1}(i))*ones(size(ddd_un,1),1);
            new_fits(:,2)=sub2ind([h,w], might_fit_row2(ddd_un(:,1)), might_fit_col2(ddd_un(:,1)));
            new_fits(:,3)=sub2ind([h,w], might_fit_row3(ddd_un(:,2)), might_fit_col3(ddd_un(:,2)));
            fits=[fits;new_fits];
        end
    end
    
    if isempty(fits)
        hull_pts=[];
        return
    end
    bad_inds=[];
    good_inds = setdiff(1:size(fits,1),bad_inds);
    ptlist=fits(good_inds,:);


    indi=0;
    hull_pts=[];
    [col_grid,row_grid] = meshgrid(1:all_cams.size_image(2),1:all_cams.size_image(1));

    for k=1:size(ptlist,1)
        c=0;
        for cam_ind=cam_inds_rec % ascending order
            c=c+1;
            PB(c,:)=all_cams.cams_array(cam_ind).invDLT*[col_grid(ptlist(k,c)),h+1-row_grid(ptlist(k,c)),1]';
        end
        ends=PB(:,1:3)./(PB(:,4));
        [P_intersect,disty] = lineIntersect3D(all_cams.all_centers_cam(:,cam_inds_rec)',ends);
        if all(disty<2e-5)
            indi=indi+1;
            hull_pts(indi,:)=P_intersect;
        end
    end
end