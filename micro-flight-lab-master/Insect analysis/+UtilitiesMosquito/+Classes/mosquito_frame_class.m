classdef mosquito_frame_class<handle
% class containing hulls and analyzed data for frame
    properties
        all_clusters; % cell array of body part clusters
        evecs; % cell array of clusters' principal component directions
        cents; % centers off clusters
        body_cent_stable; % center of hull edges
        wing_tips; % wing tip location of both wings
        body_angs; % body angles
        body_ax; % body primary vectors
        % wing edges hulls
        wing_r_front_edge;
        wing_r_back_edge;
        wing_l_front_edge;
        wing_l_back_edge;
        stroke_plane; % body axes fixed to stroke plane
        wing_angs; % wing angles
        pitches; % pitch vectors and starting points
    end
    
    methods
        function obj=mosquito_frame_class()
        % Description:
        % Constructor 
        %
        % Output:
        % obj- mosquito_frame_class
        
        end
        
        function get_body_angs(obj)
        % Description:
        % updates body angles and body axes 
        % 
        % Required input:
        % mosquito_frame - mosquito_frame_class object with this frames data

            % creat body main vector from end of tail to end of head (points up)
            headtail_cent=mean(obj.cents(3:4,:));
            head_cm=obj.all_clusters{4}-headtail_cent;
            tail_cm=obj.all_clusters{3}-headtail_cent;
            [B,I]=sort(sqrt(sum(head_cm.^2,2)));
            far_pts_head=head_cm(I(round(0.9*length(B)):end),:);
            [B,I]=sort(sqrt(sum(tail_cm.^2,2)));
            far_pts_tail=tail_cm(I(round(0.9*length(B)):end),:);
            new_body_main=mean(far_pts_head)-mean(far_pts_tail);
            body_main=(new_body_main/norm(new_body_main))';
                
            obj.body_cent_stable=0.5*(mean(far_pts_head)+mean(far_pts_tail));
            
            % get body secondary vector by using cross with the tail main vector
            % (points right)
            N = cross( obj.evecs{3}(:,1),body_main); % tail_main^body_main
            body_2=N/norm(N);
            % get body tertiary vector by using cross with the tail main vector
            % (points down)
            body_3=cross(body_main, body_2);
            obj.body_ax=[body_main,body_2,body_3];
            % get pitch from projecting main vector on xy plane
            body_main_xy=body_main;
            body_main_xy(3)=0;
            body_main_xy=body_main_xy/norm(body_main_xy);
            roti_pitch_vec=vrrotvec(body_main,body_main_xy);
            pitch_ang=rad2deg(roti_pitch_vec(4));
            roti_pitch = vrrotvec2mat(roti_pitch_vec);
            % get yaw from projecting body_main_xy vector on x axis
            roti_yaw_vec=vrrotvec(body_main_xy,[1,0,0]);
            yaw_ang=rad2deg(roti_yaw_vec(4));
            roti_yaw = vrrotvec2mat(roti_yaw_vec);
            % get roll from remaining angle between rotated body_2 and -z axis
            body_2_fix=roti_yaw*roti_pitch*body_2;
            roti_roll_vec=vrrotvec(body_2_fix,[0,0,1]);
            roll_ang=rad2deg(pi/2-roti_roll_vec(4));
            % assign output
            obj.body_angs=[pitch_ang,yaw_ang,roll_ang];
        end
        
        function get_wing_angs_edges(obj,num_of_vecs)
        % Description:
        % updates wing angles,pitches and stroke plane
        % 
        % Required input:
        % mosquito_frame - mosquito_frame_class object with this frames data
        % num_of_vecs - number of pitch angles to measure
        
            stroke_plane_ang=deg2rad(35);%%%
            rot_stroke_plane=vrrotvec2mat([obj.body_ax(:,2)',stroke_plane_ang]);
            body_stroke_plane=rot_stroke_plane*obj.body_ax;

            %%%%% fix to correct alignment:
            %%%%% main-red-forward,2-green-right,3-blue-up
            obj.stroke_plane=[body_stroke_plane(:,3),body_stroke_plane(:,2),body_stroke_plane(:,1)];
            for wing_ind=1:2
                if wing_ind==1 % right wing
                    wing_front_edge=obj.wing_r_front_edge;
                    wing_back_edge=obj.wing_r_back_edge;
                else
                    wing_front_edge=obj.wing_l_front_edge;
                    wing_back_edge=obj.wing_l_back_edge;
                end
                [~,~,e_vecs]=svd(wing_front_edge-mean(wing_front_edge),0); %center the data 
                % set main components to point outwards
                if dot(mean(wing_front_edge)-obj.cents(5,:),e_vecs(:,1))<0 
                    e_vecs(:,1)=-e_vecs(:,1);
                end
                % get elevation from projection on stroke plane
                wing_main_stroke_plane=e_vecs(:,1)-...
                    dot(e_vecs(:,1),obj.stroke_plane(:,3))*obj.stroke_plane(:,3);
                roti_elav_vec=vrrotvec(e_vecs(:,1),wing_main_stroke_plane);
                roti_elav = vrrotvec2mat(roti_elav_vec);
                elav_ang=sign(dot(e_vecs(:,1),obj.stroke_plane(:,3)))*rad2deg(roti_elav_vec(4));
                % get stroke from projecting on forward vector
                roti_stroke_vec=vrrotvec(wing_main_stroke_plane,obj.stroke_plane(:,1));
                roti_stroke = vrrotvec2mat(roti_stroke_vec);
                stroke_ang=rad2deg(roti_stroke_vec(4));
                % get pitch vectors and get angles from projecting rotated
                % pitch vectors on right/left vector
                pitch_vecs=get_vecs_for_pitch(wing_front_edge,wing_back_edge,num_of_vecs);
                pitch_ang=zeros(size(pitch_vecs,1),1);
                for yyy=1:size(pitch_vecs,1)
                    if isnan(pitch_vecs(yyy,:))
                        pitch_ang(yyy)=nan;
                    else
                        wing_2_fix=roti_stroke*roti_elav*pitch_vecs(yyy,4:6)';
                        roti_pitch1_vec=vrrotvec(wing_2_fix,(2*wing_ind-3)*obj.stroke_plane(:,2));
                        if sign(dot(wing_2_fix,obj.stroke_plane(:,3)))<0
                            pitch_ang(yyy)=360-rad2deg(roti_pitch1_vec(4));
                        else
                            pitch_ang(yyy)=rad2deg(roti_pitch1_vec(4));
                        end
                    end
                end
                obj.wing_angs(wing_ind,:)=[stroke_ang,elav_ang,nanmean(pitch_ang)];
                obj.pitches{wing_ind,1}=pitch_vecs;
                obj.pitches{wing_ind,2}=pitch_ang;
            end
        end
    end
end