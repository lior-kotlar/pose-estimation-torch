function [Fs,couples] = get_fundamental_mats(obj)
% Description:
% returns array of fundamental matrices with order F12,F13,...,F23,...
% 
% Required input:
% obj - all_cameras_class
% 
% Output:
% Fs - 3*3*number_of_couples array with fundamental matrix ordered -
% F12,F13,...,F23,...
% couples - camera indices for each fundamental matrix
    num_of_cams=length(obj.cams_array);
    couples=nchoosek(1:num_of_cams,2);

    DLT_mats = zeros(3,4,num_of_cams);
    DLT_invs = zeros(4,3,num_of_cams);
    cam_cents = zeros(3,num_of_cams);
    for cam_ind=1:num_of_cams
        DLT_mats(:,:,cam_ind)=obj.cams_array(cam_ind).reshaped_dlt;
        DLT_invs(:,:,cam_ind)=obj.cams_array(cam_ind).invDLT;
        cam_cents(:,cam_ind)=obj.cams_array(cam_ind).camera_cnt;
    end
    
    Fs = zeros(3,3,size(couples,1));
    % calculate fundamental matrices
    for couple_ind=1:size(couples,1)
        A = DLT_mats(:,:,couples(couple_ind,2))*[cam_cents(:,couples(couple_ind,1));1];
        C = [0 -A(3) A(2); A(3) 0 -A(1); -A(2) A(1) 0];% skew-symmetric matrix
        Fs(:,:,couple_ind)=C*DLT_mats(:,:,couples(couple_ind,2))*DLT_invs(:,:,couples(couple_ind,1));
    end
end