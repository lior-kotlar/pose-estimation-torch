function plot_camera_surfaces(all_cams,varargin)
% Description:
% Plots 3 planes at set distance from center.
% Planes are parallel to cameras and images are plotted on planes with
% correct scale.
% 
% Required input:
% all_cams - all_cameras_class loaded with images
%
% Optional Input:
% center - center of plane plot
% dist_from_center - distance where planes are plotted
% plane_edge_color - color of edges in plot
% plain_alpha - transperancy of images
% show_full_frame - if true, plots entire 2d frame; if false, plots the find square
% parent_axes - axes handle to plot on

    % set default values
    defaults = {zeros(3,1),1e-2,false,'none',0.6,gca};
    defaults(1:length(varargin)) = varargin;
    [center,dist_from_center,show_full_frame,plane_edge_color,...
        plain_alpha,parent_axes]=defaults{:};
    
    % loop on cameras
    for cam_ind=1:3
        % get camera parameters
        cam_dir=all_cams.Rotation_Matrix*all_cams.cams_array(cam_ind).camera_dir; % camera direction needs to be normalized
        cam_dir=cam_dir/norm(cam_dir);
        im=full(all_cams.cams_array(cam_ind).curr_im.Image);
        point_on_plane=center'-dist_from_center*cam_dir;
        if show_full_frame
            [mxr,mxc]=size(im);mnr=1;mnc=1;
        else
            [row,col]=find(im);
            mxr=max(row);mnr=min(row);mxc=max(col);mnc=min(col);
        end
        % get edgepoints 3d lines and find intersection with the plane
        show_pic=im(mnr:mxr,mnc:mxc);
        show_pic(show_pic==0)=2^16-1; % make bacground white
        pt1=[mnc,801-mxr,1]';pt2=[mnc,801-mnr,1]';pt3=[mxc,801-mxr,1]';pt4=[mxc,801-mnr,1]';
        cam_cent_rot=all_cams.Rotation_Matrix*all_cams.cams_array(cam_ind).camera_cnt;
        PB=all_cams.cams_array(cam_ind).invDLT*[pt1,pt2,pt3,pt4];
        Ps=all_cams.Rotation_Matrix*(PB(1:3,:)./PB(4,:));
        [cor1,~]=plane_line_intersect(cam_dir,point_on_plane,cam_cent_rot,Ps(:,1));
        [cor2,~]=plane_line_intersect(cam_dir,point_on_plane,cam_cent_rot,Ps(:,2));
        [cor3,~]=plane_line_intersect(cam_dir,point_on_plane,cam_cent_rot,Ps(:,3));
        [cor4,~]=plane_line_intersect(cam_dir,point_on_plane,cam_cent_rot,Ps(:,4));
        x=[cor2(1),cor4(1);cor1(1),cor3(1)];
        y=[cor2(2),cor4(2);cor1(2),cor3(2)];
        z=[cor2(3),cor4(3);cor1(3),cor3(3)];
        surface(x, y, z, ...
            'FaceColor', 'texturemap', ...
            'FaceAlpha',plain_alpha,'EdgeColor',plane_edge_color,...
            'CData', show_pic, 'CDataMapping', 'scaled','Parent',parent_axes);
    end
    colormap gray
end