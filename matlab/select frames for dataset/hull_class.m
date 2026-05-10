classdef hull_class<handle
    
    properties
        
        general
        rightwing
        leftwing
        ind2lab
        body
    end
    
    methods
        function obj = hull_class(hull_par,sp,easywand,camvec)
            % initilize hull class
            obj.rightwing.hullAndBound.hull3d = hull_par.rightwing.hull;
            obj.rightwing.hullAndBound.Tip = hull_par.params.rightwing.Tip;
            
            obj.leftwing.hullAndBound.hull3d = hull_par.leftwing.hull;
            obj.leftwing.hullAndBound.Tip = hull_par.params.leftwing.Tip;
            
            obj.body.hullAndBound.hull3d = hull_par.body.hull;
            
            obj.rightwing.vectors.span = hull_par.params.rightwing.span;
            obj.leftwing.vectors.span = hull_par.params.leftwing.span;
            obj.body.vectors.X = hull_par.params.body.Axes.X;
            
            obj.ind2lab.hull3d.realC = hull_par.realC;
            obj.general.RotMat = hull_par.RotMat;
            obj.general.RotMat_vol = hull_par.RotMat_vol;
            for k= 1:1:length(camvec)
                camnm = sprintf('cam%d',camvec(k));
                obj.general.vectors.(camnm) = (hull_par.RotMat*hull_par.camdir(:,camvec(k)))';
                %             obj.general.vectors.cam2 = (hull_par.RotMat*hull_par.camdir(:,2))';
                %             obj.general.vectors.cam3 = (hull_par.RotMat*hull_par.camdir(:,3))';
            end
            obj.general.VideoData.FirstCine = sp{1}.metaData.startFrame;
            obj.general.VideoData.FrameRate = sp{1}.metaData.frameRate;
            obj.general.VideoData.EW_coefs = easywand.coefs;
            obj.general.VideoData.imsize = [easywand.imageHeight(1),easywand.imageWidth(1)];
            
            obj.general.VideoData.sparseFrame = hull_par.sparseFrame;
            obj.general.VideoData.numOfCam = length(sp);
            obj.general.VideoData.camvec = camvec;
        end
        
        function hullReal = Hull2LabAx(obj,fr,prop1,prop2,propReal,varargin)
            % Translate indices of hull to real location in EW axes. Then, rotate
            % to lab axes
            parser = inputParser;
            addParameter(parser,'Rotate2lab',1);
            parse(parser, varargin{:})
            if parser.Results.Rotate2lab == 1
                rtmat =  obj.general.RotMat*obj.general.RotMat_vol';
            else
                rtmat = obj.general.RotMat_vol';
            end
            
            if strcmp(prop2,'Tip')
                inds = obj.(prop1).hullAndBound.(prop2)(fr,:);
                %                 rtmat =  obj.general.RotMat*obj.general.RotMat_vol';
                
            else
                inds = obj.(prop1).hullAndBound.(prop2){fr};
            end
            [hullReal] =(rtmat*[obj.ind2lab.(propReal).realC{fr,1}(inds(:,1));...
                obj.ind2lab.(propReal).realC{fr,2}(inds(:,2));...
                obj.ind2lab.(propReal).realC{fr,3}(inds(:,3))])';
        end
        % plot hull 3D and 2Dparser
        function plot3D_hull(obj,fr,propName,prop2,propReal,C,varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            parser = inputParser;
            
            addParameter(parser,'marker','.');
            addParameter(parser,'mrksz',5);
            addParameter(parser,'cmbod',0);
            parse(parser, varargin{:})
            
            
            
            hullReal = Hull2LabAx(obj,fr,(propName),(prop2),propReal)*1000;
            
            if parser.Results.cmbod == 1
                hullBod = Hull2LabAx(obj,fr,'body','hull3d','hull3d')*1000;
                
                cmbod = mean(hullBod);
                hullReal = hullReal - cmbod;
            end
            plot3(hullReal(:,1),hullReal(:,2),hullReal(:,3),'color',C,'marker',parser.Results.marker,'linestyle','none','markersize',parser.Results.mrksz);axis equal
            
        end
        function plotVecDir(obj,fr,propNameCM,prop2CM,propReal,propNameVec,propVec,C,Amp,varargin)
            parser = inputParser;
            addParameter(parser,'flipCam',0);
            addParameter(parser,'userCM',0);
            addParameter(parser,'linwd',2);
            addParameter(parser,'arrow','on');
            addParameter(parser,'linestyle','-');
            
            parse(parser, varargin{:})
            
            if length(parser.Results.userCM) > 1
                plotLoc =  parser.Results.userCM;
            else
                hullReal = Hull2LabAx(obj,fr,(propNameCM),(prop2CM),propReal)*1000;
                plotLoc = mean(hullReal);
            end
            
            if contains(propVec,'cam')
                fr = 1;
            end
            vec = obj.(propNameVec).vectors.(propVec)(fr,:)';
            if parser.Results.flipCam == 1
                vec = -vec;
            end
            quiver3(plotLoc(1),plotLoc(2),plotLoc(3),vec(1),vec(2),vec(3),Amp,'linestyle',parser.Results.linestyle,'linewidth',parser.Results.linwd,'color',C,'ShowArrowHead',parser.Results.arrow);
        end
        function lg =  plot3D_fly(obj,fr,prop2_winghull,varargin)
            hold off;
            parser = inputParser;
            
            addParameter(parser,'flipCam',0);
            addParameter(parser,'optA',0);
            addParameter(parser,'boundary',0);
            addParameter(parser,'strkpln',1);
            addParameter(parser,'optB',0);
            addParameter(parser,'optC',0);
            addParameter(parser,'title',1);
            
            addParameter(parser,'camdir',1);
            addParameter(parser,'bodyAx',1);
            addParameter(parser,'hull',1);
            addParameter(parser,'span',0);
            addParameter(parser,'normal',0);
            addParameter(parser,'cord',0);
            addParameter(parser,'cmbod',0);
            addParameter(parser,'hull2pl','hull3d');
            
            
            addParameter(parser,'hullGr','hull3d');
            
            parse(parser, varargin{:})
            propHull = parser.Results.hullGr;
            
            lg = [];
            plot3D_hull(obj,fr,'body','hull3d','hull3d','g','cmbod',parser.Results.cmbod);hold on
            Col = {'r','b'};
            Color_mat = lines(4);
            if parser.Results.hull ==1
                if length(parser.Results.optA) > 1  || length(parser.Results.optB)>1 || length(parser.Results.optC)>1
                    Col = {[0.9 0.7 0.9],[0.7 0.8 0.97]};
                end
                
                plot3D_hull(obj,fr,'rightwing',(prop2_winghull),'hull3d',Col{1},'cmbod',parser.Results.cmbod);
                plot3D_hull(obj,fr,'leftwing',(prop2_winghull),'hull3d',Col{2},'cmbod',parser.Results.cmbod);grid on;box on;
                lg = {'body','Right Wing','Left Wing'};
            end
            
            
            if length(parser.Results.optA) > 1
                lg = plotop(obj,parser.Results.optA,fr,Color_mat,parser.Results.hull,lg,'A','cmbod',parser.Results.cmbod);
            end
            if  length(parser.Results.optB)>1
                lg = plotop(obj,parser.Results.optB,fr,Color_mat,parser.Results.hull,lg,'B','cmbod',parser.Results.cmbod);
            end
            if length(parser.Results.optC) > 1
                lg = plotop(obj,parser.Results.optC,fr,Color_mat,parser.Results.hull,lg,'C','cmbod',parser.Results.cmbod);
                
            end
            
            userCM = 0;
            if parser.Results.cmbod == 1
                userCM = [0,0,0];
            end
            
            
            if parser.Results.bodyAx ==1
                Color_mat = lines(50);
                
                plotVecDir(obj,fr,'body',propHull,'hull3d','body','X',Color_mat(1,:),3,'userCM',userCM)
                plotVecDir(obj,fr,'body',propHull,'hull3d','body','Y',Color_mat(1,:),4,'userCM',userCM)
                plotVecDir(obj,fr,'body',propHull,'hull3d','body','Z',Color_mat(1,:),3,'userCM',userCM)
                lg = [lg,'X','Y','Z'];
            end
            
            if parser.Results.camdir ==1
                if ~isfield(obj.general.VideoData,'camvec')
                    obj.general.VideoData.camvec = [2 3 4]
                    
                end
                flg4cam = 0;
                for k = 1:length(obj.general.VideoData.camvec)
                    flg3cam = 0;
                    
                    camnm = sprintf('cam%d',obj.general.VideoData.camvec(k));
                    plotVecDir(obj,fr,'body',propHull,'hull3d','general',camnm,[0.3 0.3 0.3],3,'flipCam',parser.Results.flipCam,'userCM',userCM);
                    %                 plotVecDir(obj,fr,'body',propHull,'hull3d','general','cam2',[0.5 0.5 0.5],3,'flipCam',parser.Results.flipCam)
                    %                 plotVecDir(obj,fr,'body',propHull,'hull3d','general','cam3',[0.7 0.7 0.7],3,'flipCam',parser.Results.flipCam)
                    
                    lg = [lg,camnm];
                end
            end
            if parser.Results.span ~=0
                plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','span',[0.8 0.1 0.1],3,'userCM',userCM)
                plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','span',[0.1 0.1 0.8],3,'userCM',userCM)
                lg = [lg,'span','span'];
                
            end
            
            if parser.Results.normal ==1
                
                if length(parser.Results.optA) > 1
                    plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','normal_A1',[0.8 0.1 0.1],3,'userCM',userCM)
                    plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','normal_A1',[0.1 0.1 0.8],3,'userCM',userCM)
                    
                elseif  length(parser.Results.optB)>1
                    plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','normal_B1',[0.8 0.1 0.1],3,'userCM',userCM)
                    plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','normal_B1',[0.1 0.1 0.8],3,'userCM',userCM)
                else
                    
                    plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','normal',[0.8 0.1 0.1],3,'userCM',userCM)
                    plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','normal',[0.1 0.1 0.8],3,'userCM',userCM)
                end
                lg = [lg,'span','span'];
                
            end
            if parser.Results.cord ==1
                
                
                if length(parser.Results.optA) > 1
                    plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','Cord_A1',[0.8 0.1 0.1],3,'userCM',userCM)
                    plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','Cord_A1',[0.1 0.1 0.8],3,'userCM',userCM)
                    
                elseif  length(parser.Results.optB)>1
                    plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','Cord_B1',[0.8 0.1 0.1],3,'userCM',userCM)
                    plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','Cord_B1',[0.1 0.1 0.8],3,'userCM',userCM)
                else
                    
                    plotVecDir(obj,fr,'rightwing',propHull,propHull,'rightwing','Chord',[0.8 0.1 0.1],3,'userCM',userCM)
                    plotVecDir(obj,fr,'leftwing',propHull,propHull,'leftwing','Chord',[0.1 0.1 0.8],3,'userCM',userCM)
                    
                end
                lg = [lg,'Chord','Chord'];
            end
            
            if parser.Results.strkpln ==1
                plotVecDir(obj,fr,'body',propHull,propHull,'body','strkPlan',[0.8 0.1 0.1],3,'userCM',userCM)
                
            end
            
            if parser.Results.title ==1
                ttl = sprintf('frame %d',fr);
                title(ttl);%legend(lg)
            end
            
        end
        
        function [Xvec,XvecOP] = calcTimeFr(obj,angname,wingname,op,varargin)
            
            parser = inputParser;
            addParameter(parser,'time',0);
            addParameter(parser,'disterb',0);
            addParameter(parser,'marker','-*');
            addParameter(parser,'color','k');
            addParameter(parser,'linewd',1);
            addParameter(parser,'plot',1);
            addParameter(parser,'interp',0);
            addParameter(parser,'frstEnd',[0 0]);
            parse(parser, varargin{:});
            mark = parser.Results.marker;
            C = parser.Results.color;
            lnwidth = parser.Results.linewd;
            
            if sum(parser.Results.frstEnd == 0)==2
                frstEnd = [obj.general.VideoData.sparseFrame,length(obj.(wingname).angles.(angname)(:,op))+obj.general.VideoData.sparseFrame-1];
            else
                frstEnd = parser.Results.frstEnd;
            end
            
            if length(obj.(wingname).(parser.Results.prop2plot).(angname)(:,op))<length(obj.(wingname).(parser.Results.prop2plot).(angname)(op,:))
                obj.(wingname).(parser.Results.prop2plot).(angname) = obj.(wingname).(parser.Results.prop2plot).(angname)';
            end
            
            Xvec = [frstEnd(1):frstEnd(2)];
            XvecOP = Xvec;
            XvecTime=(Xvec+obj.general.VideoData.FirstCine)/obj.general.VideoData.FrameRate*1000;
            if parser.Results.time == 1
                Xvec=(Xvec+obj.general.VideoData.FirstCine)/obj.general.VideoData.FrameRate*1000;
                xlbl = 'time [ms]';
            end
        end
        
        function [XvecOP,XvecTime] =calcTimeVec(obj,varargin)
            parser = inputParser;
            addParameter(parser,'frstEnd',[0 0]);
            parse(parser, varargin{:});
            if sum(parser.Results.frstEnd == 0)==2
                frstEnd = [obj.general.VideoData.sparseFrame,length(obj.body.angles.pitch)+obj.general.VideoData.sparseFrame-1];
            else
                frstEnd = parser.Results.frstEnd;
            end
            Xvec = [frstEnd(1):frstEnd(2)];
            XvecOP = Xvec;
            XvecTime=(Xvec+obj.general.VideoData.FirstCine)/obj.general.VideoData.FrameRate*1000;
        end
        
        function [frdst1,frdst2,XvecOP,angop,XvecTime] = plotAngle(obj,angname,wingname,op,varargin)
            
            parser = inputParser;
            addParameter(parser,'time',0);
            addParameter(parser,'disterb',0);
            addParameter(parser,'marker','-*');
            addParameter(parser,'color','k');
            addParameter(parser,'linewd',1);
            addParameter(parser,'plot',1);
            addParameter(parser,'interp',0);
            addParameter(parser,'frstEnd',[0 0]);
            addParameter(parser,'addXline',0);
            addParameter(parser,'prop2plot','angles');
            
            parse(parser, varargin{:});
            mark = parser.Results.marker;
            C = parser.Results.color;
            lnwidth = parser.Results.linewd;
            frdst1 = 0;
            frdst2 = 0;
            
            if sum(parser.Results.frstEnd == 0)==2
                frstEnd = [obj.general.VideoData.sparseFrame,length(obj.(wingname).(parser.Results.prop2plot).(angname)(:,op))+obj.general.VideoData.sparseFrame-1];
            else
                frstEnd = parser.Results.frstEnd;
            end
            
            if length(obj.(wingname).(parser.Results.prop2plot).(angname)(:,op))<length(obj.(wingname).(parser.Results.prop2plot).(angname)(op,:))
                obj.(wingname).(parser.Results.prop2plot).(angname) = obj.(wingname).(parser.Results.prop2plot).(angname)';
            end
            
            Xvec = [frstEnd(1):frstEnd(2)];
            XvecOP = Xvec;
            XvecTime=(Xvec+obj.general.VideoData.FirstCine)/obj.general.VideoData.FrameRate*1000;
            xlbl = 'frame';
            if parser.Results.time == 1
                Xvec=(Xvec+obj.general.VideoData.FirstCine)/obj.general.VideoData.FrameRate*1000;
                xlbl = 'time [ms]';
            end
            
            angop = obj.(wingname).(parser.Results.prop2plot).(angname)(frstEnd(1)-obj.general.VideoData.sparseFrame+1:frstEnd(2)-obj.general.VideoData.sparseFrame+1,op);
            
            if parser.Results.plot == 1
                plot(Xvec,obj.(wingname).(parser.Results.prop2plot).(angname)(frstEnd(1)-obj.general.VideoData.sparseFrame+1:frstEnd(2)-obj.general.VideoData.sparseFrame+1,op),mark,'color',C,'linewidth',lnwidth);grid on;xlabel(xlbl);ylabel(angname);hold on
                if parser.Results.interp ~= 0
                    itrp = obj.body.idxInterp;
                    scatter(Xvec(itrp),obj.(wingname).(parser.Results.prop2plot).(angname)(itrp,op),'*r');grid on;xlabel(xlbl);ylabel(angname);hold on
                end
            end
            if parser.Results.disterb ~= 0
                
                frdst1 =    abs(obj.general.VideoData.FirstCine)-frstEnd(1)+1;
                frdst2 =    parser.Results.disterb/1000*obj.general.VideoData.FrameRate+frdst1;
                dist2 = Xvec(frdst2);
                if parser.Results.plot == 1
                    xline(dist2,'r','linewidth',3);
                    if frdst1>0
                        dist1 = Xvec(frdst1);
                        xline(dist1,'g','linewidth',3);
                    end
                end
                
            end
            if parser.Results.addXline ~= 0
                for k2 = 1:1:length(parser.Results.addXline)
                    xl = xline(Xvec(parser.Results.addXline(k2)),'b');
                    xl.LineWidth = 1;
                end
            end
            
        end
        
        function lg = plotop(obj,op,fr,Color_mat,reshull,lg,meth,varargin)
            parser = inputParser;
            addParameter(parser,'cmbod',0);
            parse(parser, varargin{:})
            
            try
                if op(2) ~=0
                    BoundnmLLE = sprintf('LE%s%d',meth,op(2));
                    BoundnmLTE = sprintf('TE%s%d',meth,op(2));
                    plot3D_hull(obj,fr,'leftwing',BoundnmLLE,'hull3d',Color_mat(2,:),'cmbod',parser.Results.cmbod);hold on;
                    plot3D_hull(obj,fr,'leftwing',BoundnmLTE,'hull3d',Color_mat(1,:),'cmbod',parser.Results.cmbod);grid on;box on;
                end
            catch
            end
            if op(1) ~=0
                
                BoundnmRLE = sprintf('LE%s%d',meth,op(1));
                BoundnmRTE = sprintf('TE%s%d',meth,op(1));
                plot3D_hull(obj,fr,'rightwing',BoundnmRLE,'hull3d',Color_mat(2,:),'cmbod',parser.Results.cmbod);hold on;
                plot3D_hull(obj,fr,'rightwing',BoundnmRTE,'hull3d',Color_mat(1,:),'cmbod',parser.Results.cmbod);hold on;
            end
            
            if reshull ==1
                lg = [lg,'Right Wing LE','Left Wing LE','Right Wing TE','Left Wing TE'];
            else
                lg = {'body','Right Wing LE','Left Wing LE','Right Wing_TE','Left Wing TE'};
            end
        end
        function plot2D(obj,fr,WingBodName,prop2_hull_mod,propRealC,hs,sp,varargin)
            % plot inverse hull on gray 2D images
            
            parser = inputParser;
            
            addParameter(parser,'plot',0);
            addParameter(parser,'Rotate2lab',1);
            addParameter(parser,'save2obj',1);
            addParameter(parser,'plotIm',1);
            addParameter(parser,'ColorPl',0);
            addParameter(parser,'marker',1);
            addParameter(parser,'plotOnlyIm',0);
            addParameter(parser,'mrksz',3);
            addParameter(parser,'DelMark',0);
            addParameter(parser,'newfr',0);
            addParameter(parser,'mrkfc',0);
            
            parse(parser, varargin{:})
            
            
            for kcam=1:1:obj.general.VideoData.numOfCam
                ind_prop = 1;
                for k = 1:1:length(WingBodName)
                    for  kprop2 = 1:1:length(prop2_hull_mod)
                        if iscell(obj.(WingBodName{k}).hullAndBound.(prop2_hull_mod{kprop2}))==0
                            coords_cell{ind_prop} = obj.(WingBodName{k}).hullAndBound.(prop2_hull_mod{kprop2})(fr,:);
                            ind_prop = ind_prop+1;
                        else
                            coords_cell{ind_prop} = obj.(WingBodName{k}).hullAndBound.(prop2_hull_mod{kprop2}){fr};
                            ind_prop = ind_prop+1;
                        end
                        
                    end
                    RealC = obj.ind2lab.(propRealC).realC(fr,:);
                end
                TwoD = Functions.Create2DCoords(RealC,obj.general.VideoData.EW_coefs(:,kcam),coords_cell,parser.Results.Rotate2lab,obj.general.VideoData.imsize(1),obj.general.RotMat_vol);
                
                if parser.Results.save2obj==1
                    ind_prop = 1;
                    for k = 1:1:length(WingBodName)
                        for  kprop2 = 1:1:length(prop2_hull_mod)
                            prop2sv = sprintf('TwoD_%s',prop2_hull_mod{kprop2});
                            obj.(WingBodName{k}).hullAndBound.(prop2sv){fr,kcam} = uint16(TwoD{ind_prop});
                            ind_prop = ind_prop+1;
                        end
                    end
                end
                if iscell(parser.Results.marker)==0
                    for kprop = 1:1:length(prop2_hull_mod)*length(WingBodName)
                        mark{kprop} = '.';
                    end
                else
                    mark = parser.Results.marker;
                end
                
                if parser.Results.plot==1
                    camname = sprintf('axcam%d',kcam);
                    
                    fullIm = Functions.ImfromSp(fr+obj.general.VideoData.sparseFrame-1,sp{kcam}.metaData,sp{kcam}.frames);
                    
                    fullImkcam = bwareafilt(fullIm>0,1);
                    [row col] = find(fullImkcam>0);
                    
                    axes(hs.(camname));
                    if  parser.Results.plotIm==1
                        if parser.Results.newfr==1
                            delete(hs.(camname).Children(1:end));
                        end
                        
                        imshow( fullIm,[]);hold on;axis equal
                        hs.(camname).XLim = [min(col)-3 max(col)+3];
                        hs.(camname).YLim = [min(row)-3 max(row)+3];
                    end
                    Color_mat = lines(length(TwoD));
                    if sum(parser.Results.ColorPl(1,:)) ~= 0
                        Color_mat = parser.Results.ColorPl;
                    end
                    if  parser.Results.plotOnlyIm==0
                        if parser.Results.DelMark==1
                            delete(hs.(camname).Children(1:end-1));
                        end
                        
                        for k2D = 1:1:length(TwoD)
                            
                            if parser.Results.mrkfc == 1
                                hold on;plot(TwoD{k2D}(:,1),TwoD{k2D}(:,2),'LineStyle','none','marker',mark{k2D},'color',Color_mat(k2D,:)...
                                    , 'markerfacecolor',Color_mat(k2D,:),...
                                    'MarkerSize',parser.Results.mrksz);
                            else
                                hold on;plot(TwoD{k2D}(:,1),TwoD{k2D}(:,2),'LineStyle','none','marker',mark{k2D},'color',Color_mat(k2D,:),...
                                    'MarkerSize',parser.Results.mrksz);
                            end
                        end
                    end
                    
                    
                    
                    axis([min(col)-3 max(col)+3 min(row)-3 max(row)+3 ]);hold on
                end
            end
        end
        function TwoDop = TwoDand_plot2D(obj,fr,WingBodName,prop2_hull_mod,propRealC,sp,varargin)
            % plot inverse hull on gray 2D images
            
            parser = inputParser;
            
            addParameter(parser,'plot',0);
            addParameter(parser,'Rotate2lab',1);
            addParameter(parser,'save2obj',0);
            addParameter(parser,'save2OP',0);
            
            
            addParameter(parser,'plotIm',1);
            addParameter(parser,'ColorPl',0);
            addParameter(parser,'marker',1);
            addParameter(parser,'plotOnlyIm',0);
            
            TwoDopm = [];
            parse(parser, varargin{:})
            
            
            
            for kcam=1:1:obj.general.VideoData.numOfCam
                ind_prop = 1;
                for k = 1:1:length(WingBodName)
                    for  kprop2 = 1:1:length(prop2_hull_mod)
                        if iscell(obj.(WingBodName{k}).hullAndBound.(prop2_hull_mod{kprop2}))==0
                            coords_cell{ind_prop} = obj.(WingBodName{k}).hullAndBound.(prop2_hull_mod{kprop2})(fr,:);
                            ind_prop = ind_prop+1;
                        else
                            coords_cell{ind_prop} = obj.(WingBodName{k}).hullAndBound.(prop2_hull_mod{kprop2}){fr};
                            ind_prop = ind_prop+1;
                        end
                        
                    end
                    RealC = obj.ind2lab.(propRealC).realC(fr,:);
                end
                TwoD = Functions.Create2DCoords(RealC,obj.general.VideoData.EW_coefs(:,kcam),coords_cell,parser.Results.Rotate2lab,obj.general.VideoData.imsize(1),obj.general.RotMat_vol);
                
                if parser.Results.save2obj==1 || parser.Results.save2OP==1
                    ind_prop = 1;
                    for k = 1:1:length(WingBodName)
                        for  kprop2 = 1:1:length(prop2_hull_mod)
                            prop2sv = sprintf('TwoD_%s',prop2_hull_mod{kprop2});
                            
                            if parser.Results.save2obj == 1
                                obj.(WingBodName{k}).hullAndBound.(prop2sv){fr,kcam} = uint16(TwoD{ind_prop});
                            end
                            if parser.Results.save2OP == 1
                                TwoDop.(WingBodName{k}).(prop2sv){fr,kcam} = uint16(TwoD{ind_prop});
                            end
                            ind_prop = ind_prop+1;
                        end
                    end
                end
                if iscell(parser.Results.marker)==0
                    for kprop = 1:1:length(TwoD)
                        mark{kprop} = '.';
                    end
                else
                    mark = parser.Results.marker;
                end
                
                if parser.Results.plot==1
                    hold off;fullIm = Functions.ImfromSp(fr+obj.general.VideoData.sparseFrame-1,sp{kcam}.metaData,sp{kcam}.frames);
                    
                    fullImkcam = bwareafilt(fullIm>0,1);
                    [row col] = find(fullImkcam>0);
                    
                    subplot(2,2,kcam)
                    if  parser.Results.plotIm==1
                        hold off;imshow(fullIm,[]);hold on
                    end
                    Color_mat = lines(length(TwoD));
                    
                    if sum(parser.Results.ColorPl(1,:)) ~= 0
                        Color_mat = parser.Results.ColorPl;
                    end
                    if  parser.Results.plotOnlyIm==0
                        for k2D = 1:1:length(TwoD)
                            hold on;plot(TwoD{k2D}(:,1),TwoD{k2D}(:,2),'LineStyle','none','marker',mark{k2D},'color',Color_mat(k2D,:));axis equal
                        end
                    end
                    axis([min(col)-3 max(col)+3 min(row)-3 max(row)+3 ]);hold on
                end
            end
        end
        
        function [frms] = UserFrmChoose(obj)
            figure; subplot(3,1,1);obj.plotAngle('pitch','body',1,'time',0);
            subplot(3,1,2);obj.plotAngle('roll','body',1,'time',0);
            subplot(3,1,3);obj.plotAngle('yaw','body',1,'time',0);
            
            figure;subplot(3,1,1);obj.plotAngle('phi','rightwing',1,'time',0,'color','r')
            hold on;obj.plotAngle('phi','leftwing',1,'time',0,'color','b')
            subplot(3,1,2);obj.plotAngle('theta','rightwing',1,'time',0,'color','r')
            hold on;obj.plotAngle('theta','leftwing',1,'time',0,'color','b')
            subplot(3,1,3);obj.plotAngle('psiA','rightwing',1,'time',0,'color','r')
            hold on;obj.plotAngle('psiA','leftwing',1,'time',0,'color','b')
            
            prompt = {'first frame:','last frame:'};
            dlgtitle = 'Input';
            dims = [1 35];
            definput = {'0','0'};
            options.WindowStyle='normal';
            answer = inputdlg(prompt,dlgtitle,dims,definput,options);
            frms(1) = str2num(answer{1});
            frms(2) =  str2num(answer{2});
        end
        
        function fixPhi(obj)
            %% fix phi values that are above 300 (TB: sometimes angles get wrong values (trigo shite) - maybe fix this)
            obj.rightwing.angles.phi(obj.rightwing.angles.phi>300) = obj.rightwing.angles.phi(obj.rightwing.angles.phi>300)-360;
            obj.leftwing.angles.phi(obj.leftwing.angles.phi>300) = obj.leftwing.angles.phi(obj.leftwing.angles.phi>300)-360;
            
            
            %% fix phi outliers (can probably be better)
            obj.rightwing.angles.phi = filloutliers(obj.rightwing.angles.phi,'center','movmedian',10);
            obj.leftwing.angles.phi  = filloutliers(obj.leftwing.angles.phi,'center','movmedian',10);
            obj.rightwing.angles.phi = filloutliers(obj.rightwing.angles.phi,'center','movmedian',10);
            obj.leftwing.angles.phi  = filloutliers(obj.leftwing.angles.phi,'center','movmedian',10);
            
        end
        
        function savehull(obj,pathofhull,mov)
            if isfield(obj.body.hullAndBound,'hull3d')
                obj.body.hullAndBound = rmfield(obj.body.hullAndBound,'hull3d');
                obj.rightwing.hullAndBound = rmfield(obj.rightwing.hullAndBound,'hull3d');
                obj.leftwing.hullAndBound = rmfield(obj.leftwing.hullAndBound,'hull3d');
            end
            filename = sprintf('hull_mov%d.mat',mov);
            dirname = sprintf('\\mov%d\\hull_op\\',mov);
            dirnameBU = sprintf('mov%d\\hull_op\\backup\\',mov);
            mkdir([pathofhull,dirnameBU])
            
            copyfile( [pathofhull,dirname,filename],[pathofhull,dirnameBU,filename],'f');
            hull = obj;
            save([pathofhull,dirname,filename],'hull')
        end
        
        
        function [anglesVec,anglesVec_dot] = SmoothWingsAngles(obj,anglesname,win2,varargin)
            parser = inputParser;
            addParameter(parser,'useSmth',0);
            parse(parser, varargin{:});
            
            % smooth the wing angles, use the smoothed data to calculate the angular
            % velocity of the wings. change signs according to the calculator axes-----
            for k = 1:1:3
                nameAng = anglesname{k};
                smmthAngName = sprintf('%s_smooth',nameAng);
                smmthAngVelName = sprintf('%s_dot_smooth',nameAng);
                for wingnm = {'rightwing','leftwing'}
                    ang2dif = fillmissing(obj.(wingnm{1}).angles.(nameAng)(:,1)*pi/180,'linear');
                    [sg0,sg1,sg2] = Functions.get_sgolay_wDeriv(ang2dif, 5, win2, 16000);
                    obj.(wingnm{1}).angles.(smmthAngName) = sg0;
                    obj.(wingnm{1}).velocity.(smmthAngVelName) = sg1;
                    
                    if parser.Results.useSmth ==1
                        anglesVec.(wingnm{1})(1:length(ang2dif),k) = sg0;
                        [sg0,sg1,sg2] = Functions.get_sgolay_wDeriv(sg0, 5, win2, 16000);
                        
                        anglesVec_dot.(wingnm{1})(1:length(sg1),k) = sg1;
                    end
                end
                
            end
            
            
        end
        
        
        function obj = BodyAxesYZ_V2(obj,angleTH,plotFlg,stfr,enfr)
            
            idx4StrkPln = ChooseSpan(obj,angleTH,0,stfr,enfr-7);
            idx4StrkPln = idx4StrkPln+stfr-1;
            
            
            while sum(diff(idx4StrkPln)<65)>0
                for k= 1:1:length(idx4StrkPln)-1
                    if idx4StrkPln(k+1)-idx4StrkPln(k)<65
                        idx4StrkPln(k+1) = [];
                        break
                    end
                end
            end
            
            %             while sum(diff(indLess65)==1)>0
            %                 indLess65 = find(diff(idx4StrkPln)<65);
            %
            %                 idx4StrkPln(find(diff(indLess65)==1))=[];
            %                 idx4StrkPln(indLess65)=[];
            %             end
            % %             indLess65 = find(diff(idx4StrkPln)<65);
            %             idx4StrkPln(indLess65)=[];
            spn_wing1 = obj.rightwing.vectors.span;
            spn_wing2 = obj.leftwing.vectors.span;
            
            % make the span direction of both wings the same
            bodAxCrosSpan = [cross([obj.body.vectors.X;obj.body.vectors.X],...
                [spn_wing1;spn_wing2])];
            FlipDirWing = dot([repmat([0,0,1],size(bodAxCrosSpan,1),1)'],bodAxCrosSpan');
            [ind] = ([FlipDirWing(1:size(FlipDirWing,2)/2)',FlipDirWing(size(FlipDirWing,2)/2+1:end)'])<0;
            spn_wing1(ind(:,1),:) = -spn_wing1(ind(:,1),:);
            spn_wing2(ind(:,2),:) = -spn_wing2(ind(:,2),:);
            for k= 1:1:length(idx4StrkPln)
                inifr = idx4StrkPln(k)-7;
                if inifr < 1
                    inifr = 1;
                end
                maxfr = idx4StrkPln(k)+7;
                if maxfr > size(obj.rightwing.hullAndBound.Tip,1) || maxfr > size(obj.leftwing.hullAndBound.Tip,1)
                    szLR = [size(obj.rightwing.hullAndBound.Tip,1),size(obj.leftwing.hullAndBound.Tip,1)];
                    maxfr =max(szLR)  ;
                end
                dotR = dot(obj.rightwing.vectors.span(inifr:maxfr,:),obj.body.vectors.X(inifr:maxfr,:),2);
                dotL = dot(obj.leftwing.vectors.span(inifr:maxfr,:),obj.body.vectors.X(inifr:maxfr,:),2);
                vecinds = [inifr:maxfr];
                [v,indR] = min(abs(dotR));
                [v,indL] = min(abs(dotL));
                
                
                
                cellmean_body = cellfun(@(x) mean(x,1),obj.body.hullAndBound.hull3d,'UniformOutput',false);
                emptCell = cell2mat(cellfun(@(x) isempty(x),cellmean_body,'UniformOutput',false))';
                cellmean_body(emptCell) = [];
                meanBod = cell2mat(cellmean_body');
                Xvec = obj.body.vectors.X;
                Xvec(emptCell,:) = [];
                
                
                TpWng1 = double([obj.rightwing.hullAndBound.Tip(emptCell~=1,:)]) - meanBod;
                TpWng2 = double([obj.leftwing.hullAndBound.Tip(emptCell~=1,:)]) - meanBod;
                TpWng = [TpWng1;TpWng2];
                rows2del = sum(abs(TpWng) >500,2);
                TpWng(rows2del==3,:) = [];
                n = obj.affine_fit(TpWng);
                
                degStrkX = real(acosd(dot(repmat(n',size(Xvec,1),1),Xvec,2)));
                if mean(degStrkX)>90
                    wakk = 3
                end
                
                
                
                
                
                
                
                
                sptmp =   [spn_wing1(vecinds(indR),:)+spn_wing2(vecinds(indL),:)]/2;
                
                sptmp = sptmp/norm(sptmp);
                
                if abs(dot(sptmp,obj.leftwing.vectors.span(vecinds(indL),:)))<0.5
                    sptmp = (obj.leftwing.vectors.span(vecinds(indL),:) - obj.rightwing.vectors.span(vecinds(indR),:));
                    sptmp = sptmp/norm(sptmp);
                end
                CheckDirY = cross(n,obj.body.vectors.X(k,:));
                CheckDirY = CheckDirY/norm(CheckDirY);
                Ybody(k,1:3) =sptmp;
                %                 if dot(n,[0,0,1])<0
                %                    n = -n;
                %                      CheckDirY = cross(n,obj.body.vectors.X(k,:));
                %                 CheckDirY = CheckDirY/norm(CheckDirY);
                %
                %                 end
                
                if dot(CheckDirY,Ybody(k,1:3))<0
                    Ybody(k,1:3) = -  Ybody(k,1:3);
                end
                
                if k>1 && dot(Ybody(k,1:3),Ybody(k-1,1:3))<0
                    Ybody(k,1:3) = -  Ybody(k,1:3);
                end
                
            end
            
            
            
            %             for k= 1:1:length(idx4StrkPln)
            %                 inifr = idx4StrkPln(k)-5;
            %                 if inifr < 1
            %                     inifr = 1;
            %                 end
            %                 maxfr = idx4StrkPln(k)+5;
            %                 if maxfr > size(obj.rightwing.hullAndBound.Tip,1) || maxfr > size(obj.leftwing.hullAndBound.Tip,1)
            %                     szLR = [size(obj.rightwing.hullAndBound.Tip,1),size(obj.leftwing.hullAndBound.Tip,1)];
            %                     maxfr =max(szLR)  ;
            %                 end
            %                 kind = 1;Tipreal=[];
            %                 for ktip = inifr:  maxfr
            %                     TiprealR{kind} = Hull2LabAx(obj,ktip,'rightwing','Tip','hull3d');
            %                     TiprealL{kind} = Hull2LabAx(obj,ktip,'leftwing','Tip','hull3d');
            %                     kind= kind+1;
            %                 end
            %                 TipCell{k} = cell2mat([TiprealR,TiprealL]');
            %
            %                 n = affine_fit(obj,TipCell{k});
            %                 Ybody(k,1:3) = cross(n,obj.body.vectors.X(k,:));
            %
            %             end
            %             save('indCell','indCell');
            spn_wing1 = obj.rightwing.vectors.span(idx4StrkPln,:);
            spn_wing2 = obj.leftwing.vectors.span(idx4StrkPln,:);
            
            % make the span direction of both wings the same
            bodAxCrosSpan = [cross([obj.body.vectors.X(idx4StrkPln,:);obj.body.vectors.X(idx4StrkPln,:)],...
                [spn_wing1;spn_wing2])];
            FlipDirWing = dot([repmat([0,0,1],size(bodAxCrosSpan,1),1)'],bodAxCrosSpan');
            [ind] = ([FlipDirWing(1:size(FlipDirWing,2)/2)',FlipDirWing(size(FlipDirWing,2)/2+1:end)'])<0;
            spn_wing1(ind(:,1),:) = -spn_wing1(ind(:,1),:);
            spn_wing2(ind(:,2),:) = -spn_wing2(ind(:,2),:);
            
            %             spn1BodAng = acosd(dot(obj.body.vectors.X(idx4StrkPln,:)',spn_wing1'));
            %             spn2BodAng = acosd(dot(obj.body.vectors.X(idx4StrkPln,:)',spn_wing2'));
            %
            %             [v,id] = min(abs([spn1BodAng',spn2BodAng']-90),[],2);
            %             sp_cell = {spn_wing1,spn_wing2};
            
            %             for k= 1:1:length(id)
            %                 sp2calcY(k,1:3) = sp_cell{id(k)}(k,:);
            %             end
            %
            
            % calculate Y axes for all picked points ( where the wings are farthest
            % away). Interpulate for the rest of the points
            %             Ybody = ([spn_wing1+spn_wing2]/2)./repmat(vecnorm([spn_wing1+spn_wing2]'/2),3,1)';
            %             Ybody = sp2calcY./(vecnorm(sp2calcY'))';
            
            Ybody = Ybody./vecnorm(Ybody,2,2);
            Ybody_inter = [interp1(idx4StrkPln,Ybody(:,1),1:1:size(obj.body.vectors.X,1),'spline')', interp1(idx4StrkPln,Ybody(:,2),1:1:size(obj.body.vectors.X,1),'spline')'...
                ,interp1(idx4StrkPln,Ybody(:,3),1:1:size(obj.body.vectors.X,1),'spline')'];
            Ybody_inter = Ybody_inter - obj.body.vectors.X .* dot(obj.body.vectors.X',Ybody_inter')'; % make Y perpendicular to Xbody
            obj.body.vectors.Y = Ybody_inter./vecnorm(Ybody_inter,2,2);
            Zbody = cross(obj.body.vectors.X,obj.body.vectors.Y);
            obj.body.vectors.Z = Zbody./vecnorm(Zbody,2,2);
            obj.body.idxInterp = idx4StrkPln;
            
        end
        
        
        % calculate body axes and angles.
        % calculate wing span and angles.
        function obj = BodyAxesYZ(obj,angleTH,plotFlg,stfr,enfr)
            
            idx4StrkPln = ChooseSpan(obj,angleTH,1,stfr,enfr-7);
            idx4StrkPln = idx4StrkPln+stfr-1;
            
            
            while sum(diff(idx4StrkPln)<65)>0
                for k= 1:1:length(idx4StrkPln)-1
                    if idx4StrkPln(k+1)-idx4StrkPln(k)<65
                        idx4StrkPln(k+1) = [];
                        break
                    end
                end
            end
            
            %             while sum(diff(indLess65)==1)>0
            %                 indLess65 = find(diff(idx4StrkPln)<65);
            %
            %                 idx4StrkPln(find(diff(indLess65)==1))=[];
            %                 idx4StrkPln(indLess65)=[];
            %             end
            % %             indLess65 = find(diff(idx4StrkPln)<65);
            %             idx4StrkPln(indLess65)=[];
            spn_wing1 = obj.rightwing.vectors.span;
            spn_wing2 = obj.leftwing.vectors.span;
            
            % make the span direction of both wings the same
            bodAxCrosSpan = [cross([obj.body.vectors.X;obj.body.vectors.X],...
                [spn_wing1;spn_wing2])];
            FlipDirWing = dot([repmat([0,0,1],size(bodAxCrosSpan,1),1)'],bodAxCrosSpan');
            [ind] = ([FlipDirWing(1:size(FlipDirWing,2)/2)',FlipDirWing(size(FlipDirWing,2)/2+1:end)'])<0;
            spn_wing1(ind(:,1),:) = -spn_wing1(ind(:,1),:);
            spn_wing2(ind(:,2),:) = -spn_wing2(ind(:,2),:);
            for k= 1:1:length(idx4StrkPln)
                inifr = idx4StrkPln(k)-7;
                if inifr < 1
                    inifr = 1;
                end
                maxfr = idx4StrkPln(k)+7;
                if maxfr > size(obj.rightwing.hullAndBound.Tip,1) || maxfr > size(obj.leftwing.hullAndBound.Tip,1)
                    szLR = [size(obj.rightwing.hullAndBound.Tip,1),size(obj.leftwing.hullAndBound.Tip,1)];
                    maxfr =max(szLR)  ;
                end
                dotR = dot(obj.rightwing.vectors.span(inifr:maxfr,:),obj.body.vectors.X(inifr:maxfr,:),2);
                dotL = dot(obj.leftwing.vectors.span(inifr:maxfr,:),obj.body.vectors.X(inifr:maxfr,:),2);
                vecinds = [inifr:maxfr];
                [v,indR] = min(abs(dotR));
                [v,indL] = min(abs(dotL));
                kind = 1;Tipreal=[];
                for ktip = vecinds(indR)-35:  vecinds(indR)+35
                    TiprealR{kind} = Hull2LabAx(obj,ktip,'rightwing','Tip','hull3d');
                    kind= kind+1;
                end
                kind = 1;
                for ktip = vecinds(indL)-35:  vecinds(indL)+35
                    ktip
                    TiprealL{kind} = Hull2LabAx(obj,ktip,'leftwing','Tip','hull3d');
                    kind= kind+1;
                end
                TipCell{k} = cell2mat([TiprealR,TiprealL]');
                indCell(k,1:2) = [vecinds(indR),vecinds(indL)];
                n = affine_fit(obj,TipCell{k});
                
                
                
                sptmp =   [spn_wing1(vecinds(indR),:)+spn_wing2(vecinds(indL),:)]/2;
                
                sptmp = sptmp/norm(sptmp);
                
                if abs(dot(sptmp,obj.leftwing.vectors.span(vecinds(indL),:)))<0.5
                    sptmp = (obj.leftwing.vectors.span(vecinds(indL),:) - obj.rightwing.vectors.span(vecinds(indR),:));
                    sptmp = sptmp/norm(sptmp);
                end
                CheckDirY = cross(n,obj.body.vectors.X(k,:));
                CheckDirY = CheckDirY/norm(CheckDirY);
                Ybody(k,1:3) =sptmp;
                if dot(n,[0,0,1])<0
                    n = -n;
                    CheckDirY = cross(n,obj.body.vectors.X(k,:));
                    CheckDirY = CheckDirY/norm(CheckDirY);
                    
                end
                
                if dot(CheckDirY,Ybody(k,1:3))<0
                    Ybody(k,1:3) = -  Ybody(k,1:3);
                end
                
                if k>1 && dot(Ybody(k,1:3),Ybody(k-1,1:3))<0
                    Ybody(k,1:3) = -  Ybody(k,1:3);
                end
                
            end
            
            
            
            %             for k= 1:1:length(idx4StrkPln)
            %                 inifr = idx4StrkPln(k)-5;
            %                 if inifr < 1
            %                     inifr = 1;
            %                 end
            %                 maxfr = idx4StrkPln(k)+5;
            %                 if maxfr > size(obj.rightwing.hullAndBound.Tip,1) || maxfr > size(obj.leftwing.hullAndBound.Tip,1)
            %                     szLR = [size(obj.rightwing.hullAndBound.Tip,1),size(obj.leftwing.hullAndBound.Tip,1)];
            %                     maxfr =max(szLR)  ;
            %                 end
            %                 kind = 1;Tipreal=[];
            %                 for ktip = inifr:  maxfr
            %                     TiprealR{kind} = Hull2LabAx(obj,ktip,'rightwing','Tip','hull3d');
            %                     TiprealL{kind} = Hull2LabAx(obj,ktip,'leftwing','Tip','hull3d');
            %                     kind= kind+1;
            %                 end
            %                 TipCell{k} = cell2mat([TiprealR,TiprealL]');
            %
            %                 n = affine_fit(obj,TipCell{k});
            %                 Ybody(k,1:3) = cross(n,obj.body.vectors.X(k,:));
            %
            %             end
            save('indCell','indCell');
            spn_wing1 = obj.rightwing.vectors.span(idx4StrkPln,:);
            spn_wing2 = obj.leftwing.vectors.span(idx4StrkPln,:);
            
            % make the span direction of both wings the same
            bodAxCrosSpan = [cross([obj.body.vectors.X(idx4StrkPln,:);obj.body.vectors.X(idx4StrkPln,:)],...
                [spn_wing1;spn_wing2])];
            FlipDirWing = dot([repmat([0,0,1],size(bodAxCrosSpan,1),1)'],bodAxCrosSpan');
            [ind] = ([FlipDirWing(1:size(FlipDirWing,2)/2)',FlipDirWing(size(FlipDirWing,2)/2+1:end)'])<0;
            spn_wing1(ind(:,1),:) = -spn_wing1(ind(:,1),:);
            spn_wing2(ind(:,2),:) = -spn_wing2(ind(:,2),:);
            
            %             spn1BodAng = acosd(dot(obj.body.vectors.X(idx4StrkPln,:)',spn_wing1'));
            %             spn2BodAng = acosd(dot(obj.body.vectors.X(idx4StrkPln,:)',spn_wing2'));
            %
            %             [v,id] = min(abs([spn1BodAng',spn2BodAng']-90),[],2);
            %             sp_cell = {spn_wing1,spn_wing2};
            
            %             for k= 1:1:length(id)
            %                 sp2calcY(k,1:3) = sp_cell{id(k)}(k,:);
            %             end
            %
            
            % calculate Y axes for all picked points ( where the wings are farthest
            % away). Interpulate for the rest of the points
            %             Ybody = ([spn_wing1+spn_wing2]/2)./repmat(vecnorm([spn_wing1+spn_wing2]'/2),3,1)';
            %             Ybody = sp2calcY./(vecnorm(sp2calcY'))';
            
            Ybody = Ybody./vecnorm(Ybody,2,2);
            Ybody_inter = [interp1(idx4StrkPln,Ybody(:,1),1:1:size(obj.body.vectors.X,1),'spline')', interp1(idx4StrkPln,Ybody(:,2),1:1:size(obj.body.vectors.X,1),'spline')'...
                ,interp1(idx4StrkPln,Ybody(:,3),1:1:size(obj.body.vectors.X,1),'spline')'];
            Ybody_inter = Ybody_inter - obj.body.vectors.X .* dot(obj.body.vectors.X',Ybody_inter')'; % make Y perpendicular to Xbody
            obj.body.vectors.Y = Ybody_inter./vecnorm(Ybody_inter,2,2);
            Zbody = cross(obj.body.vectors.X,obj.body.vectors.Y);
            obj.body.vectors.Z = Zbody./vecnorm(Zbody,2,2);
            obj.body.idxInterp = idx4StrkPln;
            
        end
        
        function [n,V,p] = affine_fit(obj,X)
            %Computes the plane that fits best (lest square of the normal distance
            %to the plane) a set of sample points.
            %INPUTS:
            %
            %X: a N by 3 matrix where each line is a sample point
            %
            %OUTPUTS:
            %
            %n : a unit (column) vector normal to the plane
            %V : a 3 by 2 matrix. The columns of V form an orthonormal basis of the
            %plane
            %p : a point belonging to the plane
            %
            %NB: this code actually works in any dimension (2,3,4,...)
            %Author: Adrien Leygue
            %Date: August 30 2013
            
            %the mean of the samples belongs to the plane
            p = mean(X,1);
            
            %The samples are reduced:
            R = bsxfun(@minus,X,p);
            %Computation of the principal directions if the samples cloud
            [V,D] = eig(R'*R);
            %Extract the output from the eigenvectors
            n = V(:,1);
            V = V(:,2:end);
        end
        
        function obj = Define_RL_wing(obj,stfr,enfr)
            % Arange wings by right and left
            
            indLT0 = find(dot(repmat([0,0,1],size(obj.body.vectors.Z,1),1)',obj.body.vectors.Z')<0);
            obj.body.vectors.Z(indLT0,:) = -obj.body.vectors.Z(indLT0,:);
            obj.body.vectors.Y = cross(obj.body.vectors.Z',obj.body.vectors.X')';
            
            
            
            Right = find(dot(obj.leftwing.vectors.span(stfr:enfr,:)',obj.body.vectors.Y(stfr:enfr,:)')'<dot(obj.rightwing.vectors.span(stfr:enfr,:)',obj.body.vectors.Y(stfr:enfr,:)')');
            spanWing1_tmp = obj.rightwing.vectors.span(stfr:enfr,:);
            spanWing2_tmp = obj.leftwing.vectors.span(stfr:enfr,:);
            
            Wing1_tmp=obj.rightwing.hullAndBound.hull3d(stfr:enfr);
            Wing2_tmp=obj.leftwing.hullAndBound.hull3d(stfr:enfr);
            
            TipWing1_tmp=obj.rightwing.hullAndBound.Tip(stfr:enfr,:);
            TipWing2_tmp=obj.leftwing.hullAndBound.Tip(stfr:enfr,:);
            
            
            obj.rightwing.vectors.span(Right+stfr-1,:) = spanWing2_tmp(Right,:);
            obj.leftwing.vectors.span(Right+stfr-1,:) = spanWing1_tmp(Right,:);
            
            obj.leftwing.hullAndBound.hull3d(Right+stfr-1)= Wing1_tmp(Right);
            obj.rightwing.hullAndBound.hull3d(Right+stfr-1)=Wing2_tmp(Right);
            
            
            obj.leftwing.hullAndBound.Tip(Right+stfr-1,:)= TipWing1_tmp(Right,:);
            obj.rightwing.hullAndBound.Tip(Right+stfr-1,:)=TipWing2_tmp(Right,:);
            clear Wing1_tmp;
        end
        function obj = bodyAngles_pitch_yaw(obj,stfr,enfr)
            % calculate body angles------------------------------------
            obj.body.angles.pitch(stfr:enfr,1) = ...
                (90 - acos(dot(  obj.body.vectors.X(stfr:enfr,:)',  repmat([0,0,1],size(obj.body.vectors.X(stfr:enfr,:),1) ,1)')  )*180/pi)';
            obj.body.angles.yaw(stfr:enfr,1) = (atan2(obj.body.vectors.X(stfr:enfr,2), obj.body.vectors.X(stfr:enfr,1))' * 180/pi)';
        end
        function [rollAng,strkPlan] = bodyAngles_roll_sp(obj,fr)
            % rotate axes to calculate roll angle
            roll_rotation_mat =  Functions.eulerRotationMatrix(obj.body.angles.yaw(fr)*pi/180, obj.body.angles.pitch(fr)*pi/180, 0) ;
            yflyZerPitch = roll_rotation_mat*obj.body.vectors.Y(fr,:)';
            
            rollAng  = atan2(yflyZerPitch(3),yflyZerPitch(2))*180/pi;
            %------------------------------------------------------
            % calculate the stroke plane - 45 deg above the X axes
            strkPlan = Functions.rodrigues_rot(obj.body.vectors.X(fr,:),obj.body.vectors.Y(fr,:),-45*pi/180);
            if dot(strkPlan,[0,0,1])<0
                strkPlan = Functions.rodrigues_rot(obj.body.vectors.X(fr,:),obj.body.vectors.Y(fr,:),45*pi/180);
            end
            obj.body.vectors.strkPlan(fr,1:3) = strkPlan;
            obj.body.angles.roll(fr,1) = rollAng;
        end
        function WingAngles_phi_theta(obj,fr,wingname,varargin)
            parser = inputParser;
            
            addParameter(parser,'spanName','span');
            addParameter(parser,'LEName','LEB1');
            addParameter(parser,'CalcLE',0);
            
            addParameter(parser,'phiName','phi');
            addParameter(parser,'thetaName','theta');
            addParameter(parser,'option',1);
            parse(parser, varargin{:})
            
            
            spanProp = parser.Results.spanName;
            LEProp = parser.Results.LEName;
            
            phiProp = parser.Results.phiName;
            thetaProp = parser.Results.thetaName;
            
            StrkplnZ = obj.body.vectors.strkPlan(fr,:);
            StrkplnY = cross(obj.body.vectors.strkPlan(fr,:),obj.body.vectors.X(fr,:));
            StrkplnX = cross(StrkplnY,obj.body.vectors.strkPlan(fr,:));
            
            spnSP = [dot(StrkplnX,obj.(wingname).vectors.(spanProp)(fr,:)),dot(StrkplnY,obj.(wingname).vectors.(spanProp)(fr,:)),dot(StrkplnZ,obj.(wingname).vectors.(spanProp)(fr,:))];
            
            
            % project span on stroke plane
            [x,y,z] = Functions.projection(obj.body.vectors.strkPlan(fr,1),obj.body.vectors.strkPlan(fr,2),obj.body.vectors.strkPlan(fr,3)...
                ,0,obj.(wingname).vectors.(spanProp)(fr,1),obj.(wingname).vectors.(spanProp)(fr,2),obj.(wingname).vectors.(spanProp)(fr,3));
            span_proj_strkpln = [x,y,z]/norm([x,y,z]);
            span_theta = obj.(wingname).vectors.(spanProp)(fr,:);
            obj.(wingname).angles.(phiProp)(fr,parser.Results.option) = 180-atan2(-spnSP(2),-spnSP(1))*180/pi;
            if strcmp(wingname,'leftwing')
                obj.(wingname).angles.(phiProp)(fr,parser.Results.option) = 180-(obj.(wingname).angles.(phiProp)(fr,parser.Results.option)-180);
            end
            
            % project body on stroke plane
            [x,y,z] = Functions.projection(obj.body.vectors.strkPlan(fr,1), obj.body.vectors.strkPlan(fr,2), obj.body.vectors.strkPlan(fr,3)...
                ,0,obj.body.vectors.X(fr,1),obj.body.vectors.X(fr,2),obj.body.vectors.X(fr,3));
            BodyAx_proj_strkpln = [x,y,z]'/norm([x,y,z]);
            
            % calculate theta and phi
            phiProp_old = sprintf('%s_old',phiProp);
            obj.(wingname).angles.(phiProp_old)(fr,parser.Results.option) = acosd( dot(BodyAx_proj_strkpln, span_proj_strkpln) );
            obj.(wingname).angles.(thetaProp)(fr,parser.Results.option)=90-real(acosd(dot(span_theta,obj.body.vectors.strkPlan(fr,:))));
            
            % project LE on stroke plane
            if parser.Results.CalcLE == 1
                
                
                LESP =  [dot(StrkplnX,obj.(wingname).vectors.(LEProp)(fr,:)),dot(StrkplnY,obj.(wingname).vectors.(LEProp)(fr,:)),dot(StrkplnZ,obj.(wingname).vectors.(LEProp)(fr,:))];
                obj.(wingname).angles.(phiProp)(fr,parser.Results.option) = 180-atan2(-spnSP(2),-spnSP(1))*180/pi;
                if strcmp(wingname,'leftwing');
                    obj.(wingname).angles.(phiProp)(fr,parser.Results.option) = 180-(obj.(wingname).angles.(phiProp)(fr,parser.Results.option)-180);
                end
                
                [x,y,z] = Functions.projection(obj.body.vectors.strkPlan(fr,1),obj.body.vectors.strkPlan(fr,2),obj.body.vectors.strkPlan(fr,3)...
                    ,0,obj.(wingname).vectors.(LEProp)(fr,1),obj.(wingname).vectors.(LEProp)(fr,2),obj.(wingname).vectors.(LEProp)(fr,3));
                LE_proj_strkpln = [x,y,z]/norm([x,y,z]);
                LEProp_theta = obj.(wingname).vectors.(LEProp)(fr,:);
                
                % calculate theta and phi
                phiProp_old = sprintf('%s_old',phiProp);
                obj.(wingname).angles.(phiProp_old)(fr,parser.Results.option) = acosd( dot(BodyAx_proj_strkpln, LE_proj_strkpln) );
                obj.(wingname).angles.(thetaProp)(fr,parser.Results.option)=90-real(acosd(dot(LEProp_theta,obj.body.vectors.strkPlan(fr,:))));
                
                
            end
            
            
        end
        function  CalcWingsVec(obj,fr,wingname,perc2Cut,prop2)
            span=obj.general.RotMat_vol*obj.general.RotMat'*obj.(wingname).vectors.span(fr,:)'; % rotate span to EW axes
            
            obj.(wingname).hullAndBound.wing_cut{fr} = Functions.RotateAndCut(double(obj.(wingname).hullAndBound.(prop2){fr}),span,perc2Cut);
            
            [spanHat,Tip,chord_dir,NormWing] = wingAnalysis(obj,wingname,prop2,fr);
            obj.(wingname).vectors.Chord(fr,:) = (obj.general.RotMat * obj.general.RotMat_vol' * chord_dir)';
            obj.(wingname).vectors.span(fr,:) = (obj.general.RotMat * obj.general.RotMat_vol' * spanHat)';
            obj.(wingname).vectors.normal(fr,:) = (obj.general.RotMat * obj.general.RotMat_vol' * NormWing)';
            obj.(wingname).hullAndBound.Tip(fr,:) = Tip;
        end
        
        
        % Wing boundaries-------------------
        function [LE,TE] = SplitLETE (obj,fr,wingname,alledges,varargin)
            % cut percentage of the wing from the root side to get a
            % cleaner wing for parameters calculations.
            % split the wing to LE and TE
            
            parser = inputParser;
            addParameter(parser,'egdehull',0);
            addParameter(parser,'LETEegde',0);
            parse(parser, varargin{:})
            
            wing_cut = obj.(wingname).hullAndBound.wing_cut{fr};
            chord_dir = obj.general.RotMat_vol*obj.general.RotMat' *obj.(wingname).vectors.Chord(fr,:)';
            Tip = obj.(wingname).hullAndBound.Tip(fr,:);
            
            if iscell(parser.Results.LETEegde)
                nmBound = strsplit(parser.Results.LETEegde{1},'LE');
                Cordname = sprintf('Cord_%s',nmBound{2});
                Tipname = sprintf('Tip_%s',nmBound{2});
                chord_dir = obj.general.RotMat_vol*obj.general.RotMat' *obj.(wingname).vectors.(Cordname)(fr,:)';
                Tip = obj.(wingname).hullAndBound.(Tipname)(fr,:);
                wing_cut = alledges;
            end
            
            
            %              if ischar(parser.Results.egdehull)
            %                 wing_cut = obj.(wingname).hullAndBound.(parser.Results.egdehull){fr};
            %              end
            
            z_tipCoords_wing=dot(chord_dir,double(Tip));
            z_wingCoords_wing=dot(repmat(chord_dir,1,size(wing_cut,1)),wing_cut');
            LE_indx_wing=z_wingCoords_wing>z_tipCoords_wing;
            
            LE=double(wing_cut(LE_indx_wing==1,:));
            TE=double(wing_cut(LE_indx_wing==0,:));
            
        end
        
        function [LE_srt,TE_srt,ind,LE_srt2d,TE_srt2d] = CreateBoundary(obj,realName,LE,TE,wing_cut,fr,SE_close_B,SE_dilate_B,Tip)
            kcamind = 1;
            for kcam=obj.general.VideoData.camvec
                TwoD = Functions.Create2DCoords(obj.ind2lab.(realName).realC(fr,1:3),obj.general.VideoData.EW_coefs(:,kcam),{LE...
                    ,TE,wing_cut,Tip,mean(wing_cut),obj.body.hullAndBound.hull3d{fr}},1,obj.general.VideoData.imsize(1),obj.general.RotMat_vol);
                [TwoDwingIm,TwoDDownIm,TwoDUpIm,Down,Up] = Split2Dbound(obj,TwoD,SE_close_B);
                % project 3d LE and TE and define the intersecting
                % pixels as LE and TE accordingaly
                BodOnW=0;LETE_same=0;
                
                if size(intersect(TwoD{3},TwoD{6},'rows'),1)>size(unique(TwoD{3},'rows'),1)*3/4
                    % if more than 75% of the wing intersects the body use
                    % the LE hull (not boundary)
                    wingLE = {LE,LE};
                    wingTE = {TE,TE};BodOnW=1;
                else
                    [wingLE,wingTE,LETE_same,TwoDwingIm,bounWing,wingLE2d,wingTE2d] = intersect2D(obj,TwoDwingIm,SE_dilate_B,LE,TE,TwoD{1},TwoD{2},TwoDUpIm,TwoDDownIm);
                end
                % Each wingLE/wingTE has 2 options, each option is constructed
                % using one of the 2D boundary. (each boundary can
                % intersect with the LE as well as the TE. usually the
                % right option will contain more pixels)
                idxs_LETE = [cell2mat(cellfun(@(x) length(x), wingLE, 'UniformOutput', false));cell2mat(cellfun(@(x) length(x), wingTE, 'UniformOutput', false))];
                [~,indM]=max([idxs_LETE(1,1)+idxs_LETE(2,2),idxs_LETE(1,2)+idxs_LETE(2,1)]);
                if indM==1
                    op_indLE = 1;op_indTE=2;
                else
                    op_indLE = 2;op_indTE=1;
                end
                % calculate the difference between amount of pixels for
                % each option. keep only the indices marked as LE or TE
                % in the 3D hull. (eventually having 3 hulls, one for
                % each camera)
                
                valDiff(kcamind)=abs(idxs_LETE(1,op_indLE)+idxs_LETE(2,op_indTE)-(idxs_LETE(1,op_indTE)+idxs_LETE(2,op_indLE)));
                LE_srt{kcamind}= [wingLE(op_indLE) wingLE(op_indTE)];
                TE_srt{kcamind}= [wingTE(op_indTE) wingTE(op_indLE)];
                LE_srt2d{kcamind}= [wingLE2d(op_indLE) wingLE2d(op_indTE)];
                TE_srt2d{kcamind}= [wingTE2d(op_indTE) wingTE2d(op_indLE)];
                [~,ind]=sort(valDiff(:),'descend');
                kcamind = kcamind + 1;
            end
        end
        
        function LETE_optio = ChooseOpt(~,LE_srt,TE_srt,ind)
            % intersect the options to identify LE and Te according to
            % size. start with the camera that has the largest difference
            % between LE and TE;
            
            kcam_size =[ind(2),ind(1)];
            kcamVec = [ind(1),ind(3)];
            for kcam_ind = 1:1:2
                for kcamVec_ind = 1:1:2
                    for optLETE = 1:1:2
                        LEinter_opt1{optLETE,kcamVec_ind,kcam_ind}=intersect(LE_srt{kcam_size(kcam_ind)}{1},LE_srt{kcamVec(kcamVec_ind)}{optLETE},'rows');
                        TEinter_opt1{optLETE,kcamVec_ind,kcam_ind}=intersect(TE_srt{kcam_size(kcam_ind)}{1},TE_srt{kcamVec(kcamVec_ind)}{optLETE},'rows');
                        
                        LEinter_opt2{optLETE,kcamVec_ind,kcam_ind}=intersect(LE_srt{kcam_size(kcam_ind)}{2},LE_srt{kcamVec(kcamVec_ind)}{optLETE},'rows');
                        TEinter_opt2{optLETE,kcamVec_ind,kcam_ind}=intersect(TE_srt{kcam_size(kcam_ind)}{2},TE_srt{kcamVec(kcamVec_ind)}{optLETE},'rows');
                        
                    end
                end
            end
            LEinter_op_c = cell(1,2);TEinter_op_c = cell(1,2);
            for k=1:1:2
                Inter_op1LE = [LEinter_opt1(1,1,1),LEinter_opt1(k,2,1),LEinter_opt1(k,2,2)]; % intersect: cam 1121 1131 2131
                Inter_op1TE = [TEinter_opt1(1,1,1),TEinter_opt1(k,2,1),TEinter_opt1(k,2,2)]; % intersect: cam 1121 1131 2131
                
                LEinter_op = intersect(Inter_op1LE{1},Inter_op1LE{2},'rows');
                TEinter_op = intersect(Inter_op1TE{1},Inter_op1TE{2},'rows');
                
                LEinter_op_c{k} = intersect(LEinter_op,Inter_op1LE{3},'rows');
                TEinter_op_c{k} = intersect(TEinter_op,Inter_op1TE{3},'rows');
            end
            
            Size_sort_LETE = [cell2mat(cellfun(@(x) length(x), LEinter_op_c, 'UniformOutput', false));cell2mat(cellfun(@(x) length(x), TEinter_op_c, 'UniformOutput', false))];
            [~,indMax] = max(sum(Size_sort_LETE));
            LETE_optio = [LEinter_op_c;TEinter_op_c];
            LETE_optio = [LETE_optio(:,indMax),LETE_optio(:,(1==indMax)*2+(2==indMax)*1)];
            
            if sum(ismember(cell2mat(LETE_optio(:,1)) , cell2mat(LETE_optio(:,2)),'rows' )) == size(cell2mat(LETE_optio(:,1)) ,1)
                for k=1:1:2
                    Inter_op2LE = [LEinter_opt2(1,1,1),LEinter_opt1(k,2,1),LEinter_opt1(k,2,2)]; % intersect: cam 1121 1131 2131
                    Inter_op2TE = [TEinter_opt2(1,1,1),TEinter_opt1(k,2,1),TEinter_opt1(k,2,2)]; % intersect: cam 1121 1131 2131
                    
                    LEinter_op = intersect(Inter_op2LE{1},Inter_op2LE{2},'rows');
                    TEinter_op = intersect(Inter_op2TE{1},Inter_op2TE{2},'rows');
                    
                    LEinter_op_inter{k} = intersect(LEinter_op,Inter_op2LE{3},'rows');
                    LEinter_op_inter{k} = intersect(TEinter_op,Inter_op2TE{3},'rows');
                    
                    if size(LEinter_op_inter{k},1)<10
                        LEinter_op2{k}= intersect(Inter_op2LE{1},Inter_op2LE{3},'rows');
                        TEinter_op2{k} = intersect(Inter_op2TE{1},Inter_op2TE{3},'rows');
                    end
                end
                
                
                Size_sort_LETE = [cell2mat(cellfun(@(x) length(x), LEinter_op2, 'UniformOutput', false));cell2mat(cellfun(@(x) length(x), TEinter_op2, 'UniformOutput', false))];
                [~,indMax] = max(sum(Size_sort_LETE));
                LETE_optio2 = [LEinter_op2;TEinter_op2];
                LETE_optio2 = [LETE_optio2(:,indMax),LETE_optio2(:,(1==indMax)*2+(2==indMax)*1)];
                
                LETE_optio = [LETE_optio(:,1),LETE_optio2(:,1)];
            end
            
            
        end
        
        function [spanHat,wingTip,chord_dir,NormWing,wingVoxels] = wingAnalysis(obj,wingname,prop2,fr,varargin)
            % calculate the wings span and tip. if stroke plane exist calculate chord and normal
            
            
            parser = inputParser;
            addParameter(parser,'cutwing',1);
            addParameter(parser,'perc2cut',0.3);
            parse(parser, varargin{:})
            
            perc2Cut = parser.Results.perc2cut;
            chord_dir = [999;999;999];
            LL = 60 * 0.65 ; % say wing length is 60. can estimate it using max(pdist(coords))
            LL2 = LL / 20 ; % / 2.5 ;
            delta = 1.1 ;
            chordFraction = 1 ;
            cBody = mean(obj.body.hullAndBound.hull3d{fr});
            
            if parser.Results.cutwing == 1
                span=obj.general.RotMat_vol*obj.general.RotMat'*obj.(wingname).vectors.span(fr,:)'; % rotate span to EW axes
                [wing_cut] = Functions.RotateAndCut(double(obj.(wingname).hullAndBound.(prop2){fr}),span,perc2Cut);
            else
                wing_cut = double(obj.(wingname).hullAndBound.(prop2){fr});
            end
            
            wingLargestCC_initial = Functions.findLargestHullCC (wing_cut);
            [farPoint, ~, list] = farthestPoint(obj,wingLargestCC_initial, cBody, LL) ;
            wingLargestCC = Functions.findLargestHullCC (wingLargestCC_initial(list,:));
            [~, ~, list2] = farthestPoint(obj,wingLargestCC, cBody, LL2) ; % list2 used for finding wing cm later
            
            [~, ~, list3] = farthestPoint(obj,wingLargestCC, cBody, LL/1.5  ) ; % list2 used for finding wing cm later
            
            
            % etimate wing center of mass (see line 487 in hullAnalysis_mk4 for a more
            % elaborate calculation
            wingCoordsForCM = wingLargestCC(list3,:) ;
            cWing = mean(wingCoordsForCM) ;
            
            spanHat = (double(farPoint) - cWing)' ;
            spanHat = spanHat / norm(spanHat) ;
            
            
            [wingTip,tipCoor] = findWingTip(obj,wingLargestCC, spanHat', cWing);
            if (isnan(wingTip(1)))
                wingTip = farPoint ;
            end
            
            wingVoxels = double(wingLargestCC) ;
            Nvox       = size(wingVoxels,1) ;
            
            mat1 = wingVoxels - repmat(cWing, Nvox,1) ;
            mat2 = repmat(spanHat', Nvox, 1) ;
            
            distFromMidSpan = abs(sum(mat1.*mat2,2) ) ;
            clear mat1 mat2
            
            % these are the voxles in the slice:
            chordRowsInd = find(distFromMidSpan<delta) ;
            
            
            if (isempty(chordRowsInd))
                % first try a larger delta
                chordRowsInd = find(distFromMidSpan<3*delta) ;
                % check if still empty
                if (isempty(chordRowsInd))
                    error('hullAnalysis:Chord','Bad clustering - empty wing chord') ;
                end
            end
            clear distFromMidSpan
            
            
            % find chord vector
            
            % among the chordRowsInd voxels, find the pair that is the most distant
            % apart. this pair defines the direction of the chord. For efficiency,
            % we exclude most of the voxels before calculating the distance matrix
            
            % caculate the distance^2 of each voxel from the wing centroid
            Nvox    = length(chordRowsInd);
            
            distVec = myNorm(obj,wingVoxels(chordRowsInd,:) - repmat(cWing, Nvox,1));
            
            
            
            % select only the top third of the voxels, i.e the most distant from
            % wing centroid. calculate chord direction using Kmeans.
            % (split the chosen voxels by their distance from each other)
            NormWing=[];
            if isfield(obj.body.vectors,'strkPlan') & isempty(obj.body.vectors.strkPlan(fr,:))==0
                [~, sortedInd] = sort(distVec,'descend') ;
                selectedInd    = chordRowsInd(sortedInd(1:ceil(Nvox*chordFraction))) ;
                
                chordGr=wingVoxels(selectedInd,:);
                
                [idx,C] = kmeans(double(chordGr),2,'Distance','sqeuclidean');
                grp_chrd={chordGr(idx==1,:),chordGr(idx==2,:)};
                mean_chrds_grpLoc=dot(obj.body.vectors.X(fr,:),mean(chordGr));
                
                dist_headChord=mean_chrds_grpLoc-[ dot(mean(grp_chrd{1}),obj.body.vectors.X(fr,:)), dot(mean(grp_chrd{2},1),obj.body.vectors.X(fr,:))];
                
                
                [~,ind_chor]=min(dist_headChord);
                TE_chrd=grp_chrd{ind_chor(1)};
                LE_chrd=grp_chrd{(ind_chor(1)==2)*1+(ind_chor(1)==1)*2};
                
                chord_dir=(mean(TE_chrd)-mean(LE_chrd))/norm(mean(TE_chrd)-mean(LE_chrd));
                
                
                if ~isempty(obj.body.vectors.strkPlan(fr,:))
                    strkplnEW = (obj.general.RotMat_vol*obj.general.RotMat'*obj.body.vectors.strkPlan(fr,:)')';
                    chordOp=[chord_dir',-chord_dir'];
                    chooseNorm=dot(chordOp,[strkplnEW;strkplnEW]');
                    chord_dir=chordOp(:,chooseNorm>0);
                end
                NormWing=cross(chord_dir,spanHat);
                NormWing=NormWing/norm(NormWing);
            end
        end
        
        function [bound] = findBfromSlc(obj,wingname,fr,delta,varargin)
            
            parser = inputParser;
            addParameter(parser,'pltslc',0);
            addParameter(parser,'pltslc2d',1);
            parse(parser, varargin{:})
            
            wingVoxels = double(obj.(wingname).hullAndBound.hull3d{fr});
            spanHat = (obj.general.RotMat_vol*obj.general.RotMat'*obj.(wingname).vectors.span(fr,:)')';
            chord_dir = (obj.general.RotMat_vol*obj.general.RotMat'*obj.(wingname).vectors.Chord(fr,:)')';
            NormWing =  (obj.general.RotMat_vol*obj.general.RotMat'*obj.(wingname).vectors.normal(fr,:)')';
            
            Nvox       = size(wingVoxels,1) ;
            mat1 = wingVoxels - repmat(mean(wingVoxels), Nvox,1) ;
            mat2 = repmat(spanHat, Nvox, 1) ;
            distFromMidSpan = (sum(mat1.*mat2,2) ) ;
            
            NumSlc = ceil((max(distFromMidSpan)-min(distFromMidSpan))/delta+5);
            ksliceind=1;
            
            for kslice = 1:1:NumSlc
                CursliceSz = max(distFromMidSpan)-delta*(kslice-1)-delta/3;
                % these are the voxles in the slice:
                chordRowsInd = find(distFromMidSpan>CursliceSz & distFromMidSpan<(CursliceSz+delta)) ;
                if (isempty(chordRowsInd))
                    continue
                end
                slcVox = wingVoxels(chordRowsInd,:);
                Ncrd = length(chordRowsInd) ;
                slice3Dcoords = mat1(chordRowsInd,:) ;
                mat2 = repmat(spanHat, Ncrd, 1) ;
                % project slice onto the plane perp to span vector
                slice2Dcoords = slice3Dcoords - sum(mat2.*slice3Dcoords,2) * spanHat ;
                clear mat2
                
                %%
                %project camera vectors onto the same plane
                kcamind = 1;
                for kcam = obj.general.VideoData.camvec
                    
                    camnm =  sprintf('cam%d',kcam);
                    cam_vec = obj.general.vectors.(camnm);
                    cam_proj(kcamind,1:3) = cam_vec - dot(cam_vec, spanHat) * spanHat ;
                    cam_proj(kcamind,1:3) = cam_proj(kcamind,1:3) / norm(cam_proj(kcamind,1:3)) ;
                    ChordProj(kcamind,1:3) =  chord_dir' - dot(chord_dir, spanHat) * spanHat' ;
                    NormProj(kcamind,1:3) =  NormWing' - dot(NormWing, spanHat) * spanHat' ;
                    
                    % find a vector on the slice plane that is vertical to each cam#_proj and
                    % to spanHat
                    p(kcamind,1:3) = cross(spanHat, cam_proj(kcamind,1:3)) ;
                    % calculate projections of slice points onto p1, p2, p3
                    slice_p_proj{kcamind} = p(kcamind,1:3) * slice2Dcoords' ;
                    % extreme for one side
                    [maxV indmaxV] = max(slice_p_proj{kcamind}) ;
                    max_minInd(kcamind,1) = indmaxV;
                    max_minVal(kcamind,1) = maxV;
                    
                    [minV indminV] = min(slice_p_proj{kcamind}) ;
                    max_minInd(kcamind,2) = indminV;
                    max_minVal(kcamind,2) = minV;
                    
                    
                    
                    distPt(kcamind) = abs(diff(dot(repmat(ChordProj(kcamind,1:3),2,1),slcVox(max_minInd(kcamind,:),:),2)));
                    distP(kcamind) = max(slice_p_proj{kcamind})-min(slice_p_proj{kcamind});
                    
                    distpval(kcamind,1:2) = [dot(repmat(ChordProj(kcamind,1:3),2,1),slice2Dcoords(max_minInd(kcamind,:)',:),2)]';
                    distNormpval(kcamind,1:2) = [dot(repmat(NormProj(kcamind,1:3),2,1),slice2Dcoords(max_minInd(kcamind,:)',:),2)]';
                    kcamind = kcamind+1;
                    
                    
                end
                
                [~,indst] = sort(distP,'descend');
                maxC = max_minInd(indst(1),:);
                Sec = max_minInd(indst(2),:);
                Pds = max_minInd(indst(3),:);
                
                distpval_reorder = distpval(indst,:);
                distNormpval_reorder = distNormpval(indst,:);
                
                mxindx = [maxC;Sec;Pds];
                
                
                
                if length(unique(max_minInd))==4 % trapezoid or makbilit
                    for k=1:1:6
                        crdNormAxall(k,1:2) = [distpval_reorder(k),distNormpval_reorder(k)];
                    end
                    [crdNormAx ia ib]  = unique(crdNormAxall,'rows');
                    crdNormAx = crdNormAx-mean(crdNormAx);
                    degcen = atan2d(-crdNormAx(:,2),crdNormAx(:,1));
                    [v,I] = sort(degcen);
                    orderCo = ia(I);
                    origUni = ia(ib);
                    
                    diagonals = [orderCo(1),orderCo(3);orderCo(2),orderCo(4)];
                    diag1 = sqrt(sum(diff(crdNormAxall(diagonals(1,:)',:)).^2));
                    diag2 = sqrt(sum(diff(crdNormAxall(diagonals(2,:)',:)).^2));
                    
                    
                    if ~ismember([origUni(1),origUni(4)],diagonals,'rows') && ~ismember([origUni(4),origUni(1)],diagonals,'rows')
                        [v idiag ]  = max([diag1,diag2]);
                        
                        reshmxmn = reshape(mxindx,6,1);
                        unire = reshmxmn(ia);
                        unireord =unire(I);
                        diagonalsInd = [unireord(1),unireord(3);unireord(2),unireord(4)];
                        maxC = diagonalsInd(idiag,:);
                    end
                end
                
                
                all_points_1mnmx = [maxC];
                cell_edge1{ksliceind} = slcVox(all_points_1mnmx,:);
                
                all_points_2mnmx = [Sec];
                cell_edge2{ksliceind} = slcVox(all_points_2mnmx,:);
                
                all_points_P = [Pds];
                cell_edge3{ksliceind} = slcVox(all_points_P,:);
                
                if parser.Results.pltslc == 1
                    
                    
                    allptsmat = [maxC;Sec;Pds]';
                    allpts = reshape(allptsmat,6,1);
                    meanslc = mean(slice2Dcoords);
                    
                    crnrs = [slice2Dcoords(allpts,:)];
                    
                    hold off;k = boundary(crnrs,0.2);
                    pl = trisurf(k,crnrs(:,1),crnrs(:,2),crnrs(:,3),'Facecolor','red','FaceAlpha',0.3);axis equal
                    
                    hold on;plot3(crnrs(1,1),crnrs(1,2),crnrs(1,3),'ro');axis equal
                    hold on;plot3(crnrs(2,1),crnrs(2,2),crnrs(2,3),'ro');axis equal
                    hold on;plot3(crnrs(3,1),crnrs(3,2),crnrs(3,3),'k*');axis equal
                    hold on;plot3(crnrs(4,1),crnrs(4,2),crnrs(4,3),'k*');axis equal
                    hold on;plot3(crnrs(5,1),crnrs(5,2),crnrs(5,3),'bs');axis equal
                    hold on;plot3(crnrs(6,1),crnrs(6,2),crnrs(6,3),'bs');axis equal
                    quiver3(meanslc(1),meanslc(2),meanslc(3),ChordProj(1,1),ChordProj(1,2),ChordProj(1,3),8);hold on
                    
                end
                ksliceind =ksliceind+1;
                
            end
            bound = {cell2mat(cell_edge1'),cell2mat(cell_edge2'),cell2mat(cell_edge3')};
        end
        
        function [psi] = CalcPsi_noB(obj,fr,wingname,varargin)
            % calculate psi
            parser = inputParser;
            addParameter(parser,'chord_name','Chord');
            addParameter(parser,'span_name','span');
            addParameter(parser,'psi_name','psi');
            parse(parser, varargin{:})
            
            psiname = parser.Results.psi_name;
            Cordname = parser.Results.chord_name;
            Spaname = parser.Results.span_name;
            
            
            Cord_dir =  obj.(wingname).vectors.(Cordname)(fr,:);
            
            if strcmp(wingname,'rightwing')
                Surf_sp=cross(obj.body.vectors.strkPlan(fr,:),obj.(wingname).vectors.(Spaname)(fr,:));
            else
                Surf_sp=cross(obj.(wingname).vectors.(Spaname)(fr,:),obj.body.vectors.strkPlan(fr,:));
            end
            
            psi=acos(dot(Cord_dir,Surf_sp))*180/pi;
            obj.(wingname).angles.(psiname)(fr) = psi;
        end
        
        
        
        % calculate psi--------------------
        function [psi] = CalcPsi(obj,fr,wingname,LE_optnm,varargin)
            % calculate psi
            parser = inputParser;
            addParameter(parser,'egdehull',0);
            parse(parser, varargin{:})
            psiname = sprintf('psi%s',LE_optnm(1));
            
            if parser.Results.egdehull == 1
                Cordname = sprintf('Cord_%s',LE_optnm);
                Spaname = sprintf('span_%s',LE_optnm);
                Normname = sprintf('normal_%s',LE_optnm);
            else
                Cordname = 'Chord';
                Spaname = 'span';
                Normname = 'normal';
            end
            
            
            
            Cord_dir =  obj.(wingname).vectors.(Cordname)(fr,:);
            
            
            
            if strcmp(wingname,'rightwing')
                Surf_sp=cross(obj.body.vectors.strkPlan(fr,:),obj.(wingname).vectors.(Spaname)(fr,:));
            else
                Surf_sp=cross(obj.(wingname).vectors.(Spaname)(fr,:),obj.body.vectors.strkPlan(fr,:));
            end
            
            
            psi=acos(dot(Cord_dir,Surf_sp))*180/pi;
            obj.(wingname).angles.(psiname)(fr-1,str2double(LE_optnm(2))) = psi;
            strkChord=acos(dot(Cord_dir,obj.body.vectors.strkPlan(fr,:)))*180/pi;
            %             if isfield(obj.(wingname).angles,psiname) == 1
            if str2double(LE_optnm(2)) == 1
                if obj.(wingname).angles.(psiname)(fr-1,str2double(LE_optnm(2)))>0 && strkChord>70
                    if abs(psi-obj.(wingname).angles.(psiname)(fr-1,str2double(LE_optnm(2))))>50
                        obj.(wingname).vectors.(Cordname)(fr,:) =  - obj.(wingname).vectors.(Cordname)(fr,:);
                        obj.(wingname).vectors.(Normname)(fr,:) =  - obj.(wingname).vectors.(Normname)(fr,:);
                        psi=180-psi;
                    end
                end
            end
            %             end
        end
        
        function [wingInEdge,spanHat,cord,normal,cWing] = CalcVec_bound(obj,LEname,allEdge,wingname,fr,varargin)
            
            
            nmBound = strsplit(LEname,'LE');
            
            Spaname = sprintf('span_%s',nmBound{2});Cordname = sprintf('Cord_%s',nmBound{2});
            Normalname = sprintf('normal_%s',nmBound{2});Tipname = sprintf('Tip_%s',nmBound{2});
            
            parser = inputParser;
            addParameter(parser,'cutWing',0.3);
            parse(parser, varargin{:})
            
            if isempty(allEdge)
                
                obj.(wingname).vectors.(Spaname)(fr,:)  = [nan nan nan];
                obj.(wingname).vectors.(Cordname)(fr,:) = [nan nan nan];
                obj.(wingname).vectors.(Normalname)(fr,:) = [nan nan nan];
                obj.(wingname).hullAndBound.(Tipname)(fr,:) = [nan nan nan];
                return
            end
            
            
            bodycr = double(obj.body.hullAndBound.hull3d{fr});
            
            labR = double(obj.(wingname).hullAndBound.hull3d{fr});
            
            
            
            [~,cutWing] = Functions.RotateAndCut(allEdge,obj.general.RotMat_vol*obj.general.RotMat'*obj.(wingname).vectors.span(fr,:)',parser.Results.cutWing);
            
            for k= 1:1:size(cutWing,1)
                distC = sqrt(sum((cutWing - cutWing(k,:)).^2,2));
                distC(distC==0)=99;
                [v I] = min(distC);
                indsort(k) = I;
                valsort(k) = v;
                
            end
            
            dellfromB = find(isoutlier(valsort,'gesd'));
            cutWing(indsort(dellfromB),:)=[];
            
            
            NormEdge = Functions.CalcNorm(cutWing);
            distfromPl = dot((labR-mean(cutWing))',(repmat(NormEdge(:,1)',size(labR,1),1))');
            indist=find((abs(distfromPl)<0.6));
            wingInEdge = labR(indist,:);
            
            cWing = mean(wingInEdge);
            LL = 60 * 0.65 ; % say wing length is 60. can estimate it using max(pdist(coords))
            L = LL / 20 ; % / 2.5 ;
            
            
            [farPoint, ~, list] =obj.farthestPoint(wingInEdge, mean(bodycr), L);
            
            spanHat = (double(farPoint) - cWing)' ;
            spanHat = spanHat / norm(spanHat) ;
            [wingTip,tipCoor] = findWingTip(obj,wingInEdge, spanHat', cWing);
            if (isnan(wingTip(1)))
                wingTip = farPoint ;
            end
            obj.(wingname).hullAndBound.(Tipname)(fr,:) = wingTip ;
            % recalculate span vector based on the refined wing tip
            spanHat = (wingTip - cWing)' ;
            obj.(wingname).vectors.(Spaname)(fr,:)  = obj.general.RotMat*obj.general.RotMat_vol'*(spanHat / norm(spanHat)) ;
            
            cord = cross(NormEdge(:,1),spanHat);
            dirStPl = dot(obj.general.RotMat*obj.general.RotMat_vol'*cord,obj.body.vectors.strkPlan(fr,:));
            if dirStPl<0
                cord = -cord;
            end
            cord = cord/norm(cord);
            obj.(wingname).vectors.(Cordname)(fr,:) = obj.general.RotMat*obj.general.RotMat_vol'*cord;
            
            normal = cross(spanHat,cord);
            obj.(wingname).vectors.(Normalname)(fr,:) = obj.general.RotMat*obj.general.RotMat_vol'*normal;
            obj.(wingname).vectors.(Normalname)(fr,:) = obj.(wingname).vectors.(Normalname)(fr,:)./norm(obj.(wingname).vectors.(Normalname)(fr,:));
            
            
            distOnSp = dot((wingInEdge-cWing)',repmat(spanHat',size(wingInEdge,1),1)',1);
            dist2use = (max(distOnSp)-min(distOnSp))/3;
            sliceCen = wingInEdge(abs(distOnSp)<(dist2use),:);
            
            
            distOnCr = dot((sliceCen-cWing)',repmat(cord',size(sliceCen,1),1)',1);
            
            slcCor = sliceCen(distOnCr>(max(distOnCr)-2),:);
            [X_cord,dist_X_cord] = find_axis(obj,slcCor);
            LEaxis = diff(X_cord)/norm(diff(X_cord));
            LEdir = dot(LEaxis,spanHat);
            if LEdir<0
                LEaxis = - LEaxis;
            end
            obj.(wingname).vectors.(LEname)(fr,:) = obj.general.RotMat*obj.general.RotMat_vol'*LEaxis';
            
            
            
            
            
            
        end
        
        function [X_cord,dist_X_cord] = find_axis(obj,X)
            % Input:
            % X - body hull: rotated to lab axes
            % plotFlag: 1/0
            
            % NOTES:
            % 1) Direction of best fit line corresponds to R(:,1)
            % 2) R(:,1) is the direction of maximum variances of dX
            % 3) D(1,1) is the variance of dX after projection on R(:,1)
            % 4) Parametric equation of best fit line: L(t)=X_ave+t*R(:,1)', where t is a real number
            % 5) Total variance of X = trace(D)
            % Coefficient of determineation; R^2 = (explained variance)/(total variance)
            N=size(X,1);
            X_ave=mean(X,1);            % mean; line of best fit will pass through this point
            dX=X-X_ave;                 % residuals
            C=(dX'*dX)/(N-1);           % variance-covariance matrix of X
            [R,D,V]=svd(C,0);           % singular value decomposition of C; C=R*D*R'
            D=diag(D);
            R2=D(1)/sum(D);
            % Visualize X and line of best fit
            % -------------------------------------------------------------------------
            % End-points of a best-fit line (segment); used for visualization only
            x=dX*R(:,1);    % project residuals on R(:,1)
            x_min=min(x);
            x_max=max(x);
            dx=x_max-x_min;
            Xa=(x_min-0.05*dx)*R(:,1)' + X_ave;
            Xb=(x_max+0.05*dx)*R(:,1)' + X_ave;
            X_cord=[Xa;Xb];
            dist_X_cord=pdist(X_cord);
        end
        
        function DelBackWing (obj,fr,wingname,tol)
            
            bodreal = obj.Hull2LabAx(fr,'body','hull3d','hull3d','Rotate2lab',1);
            winreal = obj.Hull2LabAx(fr,wingname,'hull3d','hull3d','Rotate2lab',1);
            br = mean(bodreal);
            winreal = winreal-br;
            
            disYax = dot(repmat((obj.body.vectors.Y(fr,:)),size(winreal,1),1),winreal,2);
            
            if strcmp(wingname,'rightwing') ==1
                obj.(wingname).hullAndBound.hull3d{fr} = obj.(wingname).hullAndBound.hull3d{fr}(disYax<tol,:);
            else
                obj.(wingname).hullAndBound.hull3d{fr} = obj.(wingname).hullAndBound.hull3d{fr}(disYax>tol,:);
            end
            obj.(wingname).vectors.span(fr,:) = (-mean(obj.(wingname).hullAndBound.hull3d{fr}) + double(obj.(wingname).hullAndBound.Tip(fr,:)))/...
                norm((-mean(obj.(wingname).hullAndBound.hull3d{fr}) + double(obj.(wingname).hullAndBound.Tip(fr,:))));
            obj.(wingname).vectors.span(fr,:) = obj.general.RotMat * obj.general.RotMat_vol' * obj.(wingname).vectors.span(fr,:)';
        end
        
        function [p, dst, list] = farthestPoint(~,coords, p0, L)
            % finds the point in coords which is farthest from p0
            % returns the point coordinate p and its distance from p0 in dst.
            % if there are several points with the same distance, return only one.
            % also finds the indices of voxels in coords whose distance from p is
            % smaller or equal to L
            
            % find p
            Nvox = size(coords,1) ;
            mat1 = double(coords) - repmat(p0, Nvox,1) ;
            dst2vec  = sum (mat1 .* mat1, 2) ;
            [dst, ind] = max(dst2vec) ;
            p = coords(ind,:) ;
            
            % find list
            mat1 = (coords) - repmat(p, Nvox, 1) ;
            dst2vec  = sum (mat1 .* mat1, 2) ;
            list = (dst2vec <= L^2) ;
            
        end
        
        % ---------------------------------------------------------------------
        
        function [tip,tip_coord] = findWingTip(obj,coords, spanVec, wingCM)
            
            % find distances of coords from wingCM
            Nvox = size(coords,1) ;
            
            fraction = 0.8 ;
            
            % 1st screening
            % -------------
            % consider only voxels in a cone of 30 degrees around the line starting
            % from wingCM along the direction of spanVec
            mat1 = double(coords) - repmat(wingCM, Nvox, 1) ;
            mat1 = mat1 ./ repmat( myNorm(obj,mat1), 1,3) ;
            mat2 = repmat(spanVec, Nvox, 1) ;
            dotprod = dot(mat1, mat2, 2) ;
            
            ind1 = dotprod > cosd(60) ; % cos(60 deg) = 0.5
            
            % 2nd screening
            % -------------
            % take the farthest "fraction" of the voxels and calculate their mean position
            
            coords = double(coords(ind1,:)) ;
            Nvox   = size(coords,1) ;
            
            dst  = myNorm ( obj,coords - repmat(wingCM,Nvox,1) ) ;
            
            [~, sortedInd] = sort(dst,'descend') ;
            Nvox1 = ceil(Nvox*fraction) ;
            selectedInd    = sortedInd(1:Nvox1) ;
            tip_coord = coords(selectedInd,:) ;
            
            if (isempty(selectedInd))
                tip = [NaN NaN NaN] ;
            else
                %     if (numel(selectedInd)==1)
                tip = coords(selectedInd(1),:) ;
                %     else
                %         tip = mean(coords(selectedInd,:)) ;
                %     end
            end
            
        end
        
        % ------------------------------------------------------------
        function A = myNorm(~,B)
            A = vecnorm(B,2,2) ;
        end
        
        function [TwoDwingIm,TwoDDownIm,TwoDUpIm,Down,Up] = Split2Dbound(obj,TwoD,SE)
            % Generate the boundary of the projected hull and split it.
            % make sure there are no holes in the projected image by
            % closeing it (SE)
            % TwoD contains the projection of LE,TE,wing_cut,Tip,Wing
            % CM,body.
            
            
            span2D=diff([TwoD{5};TwoD{4}])/norm([TwoD{5};TwoD{4}]); % calculate the 2D span
            slp=1./(span2D/span2D(2));
            b=TwoD{5}(2)-TwoD{5}(1)*slp(1);
            wingAll=[TwoD{1};TwoD{2}];
            if isinf(slp(1))
                yperp_wall=(ones(1,length(wingAll(:,1)))*TwoD{5}(2))';
            else
                yperp_wall=slp(1)*wingAll(:,1)+b;
            end
            % split the image according to span
            Down=wingAll(wingAll(:,2)<yperp_wall,:);
            Up=wingAll(wingAll(:,2)>=yperp_wall,:);
            
            TwoDwingIm=Functions.generateIm(obj.general.VideoData.imsize(1),obj.general.VideoData.imsize(2),TwoD{3});
            TwoDDownIm=Functions.generateIm(obj.general.VideoData.imsize(1),obj.general.VideoData.imsize(2),Down);
            TwoDDownIm=imclose(TwoDDownIm,SE);
            
            TwoDUpIm=Functions.generateIm(obj.general.VideoData.imsize(1),obj.general.VideoData.imsize(2),Up);
            TwoDUpIm=imclose(TwoDUpIm,SE);
            
            TwoDwingIm=imclose(TwoDwingIm,SE);
        end
        
        function [wingLE,wingTE,LETE_same,TwoDwingIm,bounWing,wingLE2d,wingTE2d,Boundary_wing] = intersect2D(~,TwoDwingIm,SE,LE_wing,TE_wing,TwoDwingLE,TwoDwingTE,Bup,Bdown)
            % Create the boundary of the wing from the image and dilate it
            LETE_same=0;
            [wing_coordsy wing_coordsx]=find(TwoDwingIm);
            CropedIm=TwoDwingIm(min(wing_coordsy):max(wing_coordsy),min(wing_coordsx):max(wing_coordsx));
            CropedIm=imclose(CropedIm,SE);
            bounWing_dil = bwperim(CropedIm);
            %             bounWing{1}=imdilate(bounWing_dil,SE).*Bdown(min(wing_coordsy):max(wing_coordsy),min(wing_coordsx):max(wing_coordsx));
            %             bounWing{2}=imdilate(bounWing_dil,SE).*Bup(min(wing_coordsy):max(wing_coordsy),min(wing_coordsx):max(wing_coordsx));
            
            bounWing{1}=bounWing_dil.*Bdown(min(wing_coordsy):max(wing_coordsy),min(wing_coordsx):max(wing_coordsx));
            bounWing{2}=bounWing_dil.*Bup(min(wing_coordsy):max(wing_coordsy),min(wing_coordsx):max(wing_coordsx));
            
            
            for k=1:1:2
                % count the amount of intersecting pixels for each boundary
                % with the projection of LE and TE
                [boundRo,boundCo]=find(bounWing{k});
                Boundary_wing{k}=[boundCo+min(wing_coordsx)-1,boundRo+min(wing_coordsy)-1];
                
                ibLE=(ismember(TwoDwingLE,Boundary_wing{k},'rows'));
                ibTE=(ismember(TwoDwingTE,Boundary_wing{k},'rows'));
                wingLE{k}=(LE_wing(ibLE,:)')';
                wingTE{k}=(TE_wing(ibTE,:)')';
                
                wingLE2d{k}=(TwoDwingLE(ibLE,:)')';
                wingTE2d{k}=(TwoDwingTE(ibTE,:)')';
                
                
                if abs(sum(ismember(TwoDwingLE,TwoDwingTE,'rows')))>size(TwoDwingLE,1)*4/5
                    LETE_same = 1;
                end
            end
        end
        
        function estPsi(obj)
            wingRL = {'rightwing','leftwing'};
            for kwing = 1:1:2
                wingname = (wingRL{kwing});
                
                obj.(wingname).angles.psiA(obj.(wingname).angles.psiA == 0 ) = nan;
                obj.(wingname).angles.psiB(obj.(wingname).angles.psiB == 0 ) = nan;
                
                if size(obj.(wingname).angles.psiA,1)<size(obj.(wingname).angles.psiB,1)
                    tmpA = nan(size(obj.(wingname).angles.psiB));
                    tmpA(1:size(obj.(wingname).angles.psiA,1),:) = obj.(wingname).angles.psiA;
                    obj.(wingname).angles.psiA = tmpA;
                end
                if size(obj.(wingname).angles.psiB,1)<size(obj.(wingname).angles.psiA,1)
                    tmpB = nan(size(obj.(wingname).angles.psiA));
                    tmpB(1:size(obj.(wingname).angles.psiB,1),:) = obj.(wingname).angles.psiB;
                    obj.(wingname).angles.psiB = tmpB;
                end
                obj.(wingname).angles.psiA(isnan(obj.(wingname).angles.psiA)) = obj.(wingname).angles.psiB(isnan(obj.(wingname).angles.psiA));
                obj.(wingname).angles.psiB(isnan(obj.(wingname).angles.psiB)) = obj.(wingname).angles.psiA(isnan((obj.(wingname).angles.psiB)));
                
                
                
                obj.(wingname).angles.psi = (obj.(wingname).angles.psiA+ obj.(wingname).angles.psiB)/2;
                obj.(wingname).angles.psi(isnan(obj.(wingname).angles.psi)) = 0;
            end
            
        end
        
        function [Xvec,tms] = GetTimeVec(obj)
            Xvec = [obj.general.VideoData.sparseFrame:1:...
                length(obj.body.hullAndBound.hull3d)+obj.general.VideoData.sparseFrame-1];
            tms = (Xvec+obj.general.VideoData.FirstCine)/obj.general.VideoData.FrameRate*1000;
            obj.general.VideoData.frVec = Xvec;
            obj.general.VideoData.tmVec = tms;
        end
        
        function [meanData,frmMeanData,tmMeanData] = strkMeanSim(obj,data2mean,bodyWing,prop,varargin)
            mnmaxfrm = sort([obj.rightwing.StrkMean.wingsMin(:,1);obj.rightwing.StrkMean.wingsMax(:,1)]);
            
            for k = 1:1:size(mnmaxfrm,1)-1
                meanData(k,:) = mean(data2mean(mnmaxfrm(k):mnmaxfrm(k+1),:));
                frmMeanData(k) = ((obj.general.VideoData.frVec(mnmaxfrm(k))+obj.general.VideoData.frVec(mnmaxfrm(k+1))))/2;
                tmMeanData(k) = ((obj.general.VideoData.tmVec(mnmaxfrm(k))+obj.general.VideoData.tmVec(mnmaxfrm(k+1))))/2;
                
            end
            obj.(bodyWing).StrkMean.mnmaxfrm = frmMeanData;
            obj.(bodyWing).StrkMean.mnmaxtime = tmMeanData;
            obj.(bodyWing).StrkMean.(prop) = meanData;
        end
        
        function plotStrkmean(obj,bodyWing,prop,ylbl,varargin)
            parser = inputParser;
            addParameter(parser,'time',1);
            addParameter(parser,'marker','*');
            addParameter(parser,'Oneplot',0);
            parse(parser, varargin{:})
            Xvec = obj.(bodyWing).StrkMean.mnmaxtime;
            
            if parser.Results.time == 0
                Xvec = obj.(bodyWing).StrkMean.mnmaxfrm;
            end
            if parser.Results.Oneplot ~= 0
                plot(Xvec,obj.(bodyWing).StrkMean.(prop)(:, parser.Results.Oneplot),'marker',parser.Results.marker);grid on;xlabel('time [ms]');ylabel(ylbl{1})
            else
                ax1 = subplot(3,1,1);plot(Xvec,obj.(bodyWing).StrkMean.(prop)(:,1),'marker',parser.Results.marker);grid on;xlabel('time [ms]');ylabel(ylbl{1})
                ax2 = subplot(3,1,2);plot(Xvec,obj.(bodyWing).StrkMean.(prop)(:,2),'marker',parser.Results.marker);grid on;xlabel('time [ms]');ylabel(ylbl{2})
                ax3 = subplot(3,1,3);plot(Xvec,obj.(bodyWing).StrkMean.(prop)(:,3),'marker',parser.Results.marker);grid on;xlabel('time [ms]');ylabel(ylbl{3})
                linkaxes([ax1,ax2,ax3],'x')
            end
        end
        %-------------------------------------------------------------
        function WingAmpX0_phi(obj,varargin)
            parser = inputParser;
            addParameter(parser,'addTime',0);
            addParameter(parser,'SimMean_name',0);
            addParameter(parser,'SimPropName','torque');
            addParameter(parser,'BodyWingName','body');
            
            
            parse(parser, varargin{:})
            wingnm = {'rightwing','leftwing'};
            axsnmPhi = {'axphiR','axphiL'};
            clr = {'r','b'};
            
            for kwng = 1:1:2
                % find location of peaks of phi-----------
                obj.(wingnm{kwng}).angles.phi(obj.(wingnm{kwng}).angles.phi>300) = obj.(wingnm{kwng}).angles.phi(obj.(wingnm{kwng}).angles.phi>300)-360;
                filmsiss = fillmissing(obj.(wingnm{kwng}).angles.phi,'movmedian',10);
                
                [cm Vx] = Functions.my_sgolay_smooth_and_diff(filmsiss, 5, 21, 16000, 0);
                [pksMx locsMx] = findpeaks(cm,'MinPeakDistance',50);
                [pksRMn locsMn] = findpeaks(-cm,'MinPeakDistance',50);
                %----------------------
                % find value of phi in max/min location------
                maxphi_real = obj.(wingnm{kwng}).angles.phi(locsMx);
                minphi_real = obj.(wingnm{kwng}).angles.phi(locsMn);
                %--------------------------------------------
                
                xsame = min([length(pksRMn),length(pksMx)]);
                Amp = [-minphi_real(1:xsame) + maxphi_real(1:xsame) ;-minphi_real(1:xsame-1) + maxphi_real(2:xsame)];
                
                
                obj.(wingnm{kwng}).StrkMean.wingsMax =  [locsMx(1:xsame),maxphi_real(1:xsame)];
                obj.(wingnm{kwng}).StrkMean.wingsMin = [locsMn(1:xsame),minphi_real(1:xsame)] ;
                
                
                % find amplitude and X0 of each half stroke--------------
                FrmsAmp1 = (locsMn(1:xsame)+locsMx(1:xsame))/2;
                FrmsAmp2 = (locsMn(1:xsame-1)+locsMx(2:xsame))/2;
                
                [v I] = sort([FrmsAmp1;FrmsAmp2]);
                obj.(wingnm{kwng}).StrkMean.Amp = Amp(I);
                obj.(wingnm{kwng}).StrkMean.Frms = v;
                
                X0 = [(minphi_real(1:xsame) + maxphi_real(1:xsame))/2;(minphi_real(1:xsame-1) + maxphi_real(2:xsame))/2];
                delfrmmn = 30;
                delfrmmx = 30;
                for k = 1:1:length(X0)
                    if (v(k)-delfrmmn)<1
                        delfrmmn = v(k)-1;
                    end
                    if (v(k)+delfrmmx)>=length(obj.(wingnm{kwng}).angles.phi)
                        delfrmmx = length(obj.(wingnm{kwng}).angles.phi)-1-v(k);
                    end
                    % find the frame with the closest value to X0
                    [vmin ind] = min(abs(obj.(wingnm{kwng}).angles.phi(round(v(k)-delfrmmn):floor(v(k)+delfrmmx))-X0(k)));
                    obj.(wingnm{kwng}).StrkMean.Frms_ClosestX0(k) = ind+v(k)-delfrmmn;
                end
                
                obj.(wingnm{kwng}).StrkMean.X0 = X0(I);
                
                
                tmvec = obj.general.VideoData.tmVec;
                Tm12 =  [(tmvec(locsMn(1:xsame))+tmvec(locsMx(1:xsame)))/2,(tmvec(locsMn(1:xsame-1))+tmvec(locsMx(2:xsame)))/2];
                obj.(wingnm{kwng}).StrkMean.Tm12 = Tm12(I);
                
            end
        end
        
        
        function GetTrendline(obj,varargin)
            parser = inputParser;
            addParameter(parser,'smoothpar',0.1);
            parse(parser, varargin{:});
            
            obj.rightwing.StrkMean.phitrend = smooth(obj.rightwing.angles.phi_old,parser.Results.smoothpar,'loess');
            obj.leftwing.StrkMean.phitrend = smooth(obj.leftwing.angles.phi_old,parser.Results.smoothpar,'loess');
            
            
        end
    end
    
    
    methods (Access = private)
        % private functions used to calculate body axes and angles and wing
        % angles and span.
        function idx4StrkPln = ChooseSpan(obj,angleTH,plotFlg,stfr,enfr)
            % choose the frames used to calculate the Y axis. In those
            % frames the wings are the farthest apart
            
            % projection of each wing span on body axis
            dotspanAx_wing1=dot(obj.rightwing.vectors.span(stfr:enfr,:)', obj.body.vectors.X(stfr:enfr,:)');
            dotspanAx_wing2=dot(obj.leftwing.vectors.span(stfr:enfr,:)', obj.body.vectors.X(stfr:enfr,:)');
            
            dotspanAx_wing2 = filloutliers(dotspanAx_wing2,'pchip');
            dotspanAx_wing1 = filloutliers(dotspanAx_wing1,'pchip');
            
%              dotspanAx_wing1(abs(dotspanAx_wing2)>10) = [];
%             dotspanAx_wing2(abs(dotspanAx_wing2)>10)=[];
            
            % Calculate and choose the greatest angles (between both wings)
            distSpans=(acosd(dot(obj.rightwing.vectors.span(stfr:enfr,:)',obj.leftwing.vectors.span(stfr:enfr,:)')));
            distSpans = real(distSpans);
            angBodSp=(acosd(dot(obj.rightwing.vectors.span(stfr:enfr,:)',obj.body.vectors.X(stfr:enfr,:)')));
            angBodSp = real(angBodSp);
            
            mean_strks=mean(([dotspanAx_wing1;dotspanAx_wing2]));
            changeSgn=[mean_strks<0;mean_strks>=0];
            
            [FrbckStrk] = FindUp_downStrk(obj,changeSgn,mean_strks,0);
            [idx4StrkPln] = Choose_GrAng_wing1_wing2(obj,distSpans,FrbckStrk,angleTH,10);
            
            idx4StrkPln=unique(idx4StrkPln);
            idx4StrkPln(idx4StrkPln<1)=[];
            idx4StrkPln(abs(angBodSp(idx4StrkPln)-90)>20) = [];
            if plotFlg==1
                figure;
                plot(mean_strks);hold on;grid on;title('Projection of mean span on body axis');xlabel('frames');ylabel('angle [deg]')
                plot(idx4StrkPln,mean_strks(idx4StrkPln),'*r')
                plot(idx4StrkPln,mean_strks(idx4StrkPln),'*k')
                
                figure;
                plot(distSpans);hold on;grid on;title('angle between spans');xlabel('frames');ylabel('angle [deg]')
                plot(idx4StrkPln,distSpans(idx4StrkPln),'*r')
                plot(idx4StrkPln,distSpans(idx4StrkPln),'*k')
            end
            
        end
        function [idx4StrkPln] = Choose_GrAng_wing1_wing2(~,distSpans,FrbckStrk,angleTH,Search_cell)
            % make sure the chosen point are indeed the greatest. Check +/- 10 cells
            % around the chosen index
            
            idx4StrkPln = FrbckStrk(distSpans(FrbckStrk)>angleTH);
            
            for kind_strk=1:1:length(idx4StrkPln)
                inds2ch=idx4StrkPln(kind_strk)-Search_cell:idx4StrkPln(kind_strk)+Search_cell;
                inds2ch(inds2ch<=0)=[];
                inds2ch(inds2ch>length(distSpans))=[];
                [val_max indMax]=max(distSpans(inds2ch));
                idx4StrkPln(kind_strk)=inds2ch(indMax);
            end
        end
        function [idxdwnStrk] = FindUp_downStrk(~,changeSgn,mean_strks,up_down_strk)
            % find where the signs change (compared to 0 on the body axis).
            % in thhis position the wing are farthest apart.
            
            downstrk=find((changeSgn(1,1:end-1)+changeSgn(2,2:end))==up_down_strk);
            mean_val=[mean_strks(downstrk+1);mean_strks(downstrk)];
            [~,indMin]=min(abs(mean_val),[],1);
            idx_vec=[downstrk+1;downstrk];
            idxdwnStrk=(idx_vec([(indMin==1);(indMin==2)]));
        end
        
        % private functions used to generate wing boundaries
        
    end
end




















