function  keepMoveFile_time(time,camname,k,pathOrig,movdir,path2,camTime,kcam,movnew,movdirnew)
 comp1_toOther = sum(camTime{1}(1:4)==camTime{kcam}(1:4))==4;
if comp1_toOther==1
            movname_cam=sprintf('mov%d_%s_sparse.mat',time{k,1},camname);
            movname_camNew = sprintf('mov%d_%s_sparse.mat',movnew,camname);

            copyfile([pathOrig,movdir,movname_cam],[path2,movdirnew,movname_camNew],'f')   
else           
%     error('need to check this part of the code!')
            for mov_ind=1:1:size(time,1)
                if isempty(time{mov_ind,kcam+1})==0
                    cam5Time = str2double(regexp(time{mov_ind,kcam+2}, '\d+', 'match'));
                    comp1_toOther=sum(camTime{1}(1:4)==cam5Time(1:4))==4;
                    if comp1_toOther==1
                        movname_cam=sprintf('mov%d_%s_sparse.mat',time{mov_ind,1},camname);
                        movdir_tmp=sprintf('mov%d\\',time{mov_ind,1});
                        movname_cam3_tmp=sprintf('mov%d_%s_sparse.mat',movnew,camname);
                        
                        copyfile([pathOrig,movdir_tmp,movname_cam],[path2,movdirnew,movname_cam3_tmp],'f');
                    end
                end
                
            end
        end
end

