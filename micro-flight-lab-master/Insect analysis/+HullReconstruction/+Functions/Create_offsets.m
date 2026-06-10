function [ offset_list ] = Create_offsets(row_col_size)
% Description:
% 3D matrix of neighboors of the examined pixel, the matrix is arranged as
% a spiral; used if the original seed is dark
% 
% Required input:
% row_col_size - size of volume around seed to be searched for a lit seed 
% 
% Output:
% offset_list - 3D matrix of neighboors of the examined pixel, the matrix is arranged as
% a spiral

    len_off=row_col_size^3;
    vec_offsets=(1:row_col_size)-ceil(row_col_size/2);
    % define the movement in coordinates X,Y,Z
    ind_offsetsx=reshape(repmat(vec_offsets,row_col_size^2,1),1,len_off);
    ind_offsetsy=reshape((repmat(vec_offsets,row_col_size,row_col_size)),1,len_off);
    ind_offsetsz=repmat(vec_offsets',row_col_size^2,1)';

    ind_offsets=[ind_offsetsx;ind_offsetsy;ind_offsetsz];
    ind_offsets(:,ceil(len_off/2))=[]; % remove the [0,0,0] component
    % arange the coordinates so that the closest pixels will be located first
    [~,I]=sort(sum(ind_offsets.^2,1));
    offset_list=ind_offsets(:,I);
end