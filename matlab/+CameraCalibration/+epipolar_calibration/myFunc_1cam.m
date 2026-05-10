function f = myFunc_1cam(dlt_2_11,fix_cam_ind,other_dlts, pts)
    num_of_cams=size(other_dlts,2);
    DLT_mats = zeros(3,4,num_of_cams);
    DLT_invs = zeros(4,3,num_of_cams);
    cam_cents = zeros(3,num_of_cams);
    for cam_ind=1:num_of_cams
        if cam_ind==fix_cam_ind
            %10
            dlts(2:11,cam_ind)=dlt_2_11(:);
            coefs1=dlts(:,cam_ind);
            dlts(1,cam_ind)=-((coefs1(2)*coefs1(10) + coefs1(3)*coefs1(11))*(coefs1(5)*coefs1(9) + ...
                coefs1(6)*coefs1(10) + coefs1(7)*coefs1(11)) - (coefs1(2)*coefs1(6) + ...
                coefs1(3)*coefs1(7))*(coefs1(9)^2 + coefs1(10)^2 + coefs1(11)^2))/ ...
                (coefs1(9)*(coefs1(5)*coefs1(9) + coefs1(6)*coefs1(10) + coefs1(7)*coefs1(11)) ...
                - coefs1(5)*(coefs1(9)^2 + coefs1(10)^2 + coefs1(11)^2));
        else
            dlts(:,cam_ind)=other_dlts(:,cam_ind);
        end
        [xyz,~,~,~,~,~] = CameraCalibration.epipolar_calibration.DLTcameraPosition(dlts(:,cam_ind));
        newDLT=reshape([dlts(:,cam_ind);1],[4,3])';

        DLT_mats(:,:,cam_ind)=newDLT;
        DLT_invs(:,:,cam_ind)=pinv(newDLT);
        cam_cents(:,cam_ind)=xyz;
    end
    couples=nchoosek(1:num_of_cams,2);
    
    [row,~]=find(couples==2);
    
    % calculate fundamental matrices
    f=0;
%     for couple_ind=1:size(couples,1)
    for couple_ind=row'
        A = DLT_mats(:,:,couples(couple_ind,2))*[cam_cents(:,couples(couple_ind,1));1];
        C = [0 -A(3) A(2); A(3) 0 -A(1); -A(2) A(1) 0];% skew-symmetric matrix
        F_mat=C*DLT_mats(:,:,couples(couple_ind,2))*DLT_invs(:,:,couples(couple_ind,1));

        for pt_ind=1:size(pts,2)
            f=f+(squeeze(pts(couples(couple_ind,2),pt_ind,:))'...
                *F_mat*squeeze(pts(couples(couple_ind,1),pt_ind,:)))^2;
        end
    end
    f=f/num_of_cams/size(pts,2);
end
