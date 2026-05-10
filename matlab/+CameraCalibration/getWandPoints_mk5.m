function M = getWandPoints_mk5 (radius1, radius2, cam1Filename, cam2Filename, cam3Filename, outputCSVfile, skip)
% find the centers of the two spheres that make up the "easy wand"
% calibrator.
%
% input parameters:
%    *  radius1, radius2 - radii (range?) of the spheres in pixels. if
%       inputs parameters are empty, set to default values of 17 and 45.
%
%    *  cam1Filename, cam2Filename, cam3Filename - input cine file names. 
%
%    *  outputCSVfile - name of the file to output points to as a CSV.
%       file extension is not needed. The has 12 columns
%
%    * skip - the number of frames to skip when going over the movie

%       cam1pt1x cam1pt1y cam2pt1x cam2pt1y cam3pt1x cam3pt1y    cam1pt2x cam1pt2y cam2pt2x cam2pt2y cam3pt2x cam3pt2y
%
%
%  output parameters:
%       M - the coordinates of the points in the same format as the CSV file.
%
%
% if using cin files as inputs, the following two commands need to be
% called before running this function:
%
% 1. set phantom SDK path by executing the script:
% phantomSDK_setPath
% 2. Load the phantom SDK libraries using the function:
% LoadPhantomLibraries()

% spmd
%     LoadPhantomLibraries
% end

%%%%%%%%%%%%%%%%%%%% usage example(default values): %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------
% M = CameraCalibration.getWandPoints_mk5(50, 110,...
%     'D:\Video\2019_04_17_magnet_mosquito\calibration (from subsequent day)\mov1_cam2.cine',...
%     'D:\Video\2019_04_17_magnet_mosquito\calibration (from subsequent day)\mov1_cam3.cine',...
%     'D:\Video\2019_04_17_magnet_mosquito\calibration (from subsequent day)\mov1_cam4.cine',...
%     'wand_data1_18_04_2019', 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (isempty(radius1))
    radius1 = 17 ;
end

if (isempty(radius2))
    radius2 = 45 ;
end

isTif = strcmp(cam1Filename(end-3:end),'.tif') ;
isCin = strcmp(cam1Filename(end-3:end),'.cin') || strcmp(cam1Filename(end-4:end),'.cine');

if (isTif)
%{
    info1=imfinfo(cam1Filename);
    info2=imfinfo(cam3Filename);
    info3=imfinfo(cam2Filename);
    if (length(info1)~=length(info2) || length(info1)~=length(info3))
        error('movies should have the same time axis. check this.') ;
    end
    Nim = length(info1) ;
    t01 = [] ;
    t02 = [] ;
    t03 = [] ;
    md1 = [] ;
    md2 = [] ;
    md3 = [] ;
    
    height1 = info1.Height ;
    height2 = info2.Height ;
    height3 = info3.Height ;
    
    width1 = info1.Width ;
    width2 = info2.Width ;
    width3 = info3.Width ;
%}
    error('movie must be cine. sorry.');
    
elseif (isCin) % read cin metadata
    md1 = getCinMetaData(cam1Filename) ;
    md2 = getCinMetaData(cam2Filename) ;
    md3 = getCinMetaData(cam3Filename) ;
    
    % open cin files
    md1.cindata = myOpenCinFile(md1.filename) ;
    md2.cindata = myOpenCinFile(md2.filename) ;
    md3.cindata = myOpenCinFile(md3.filename) ;
    
    height1 = md1.height ;
    height2 = md2.height ;
    height3 = md3.height ;
    
    width1 = md1.width ;
    width2 = md2.width ;
    width3 = md3.width ;
    
    t01 = md1.firstImage ;
    t02 = md2.firstImage ;
    t03 = md3.firstImage ;
    
    info1 = [] ;
    info2 = [] ;
    info3 = [] ;
    
    if (md1.firstImage~=md2.firstImage || md1.firstImage~=md3.firstImage)
        error('movies should have the same time axis. check this.') ;
    end
    Nim = md1.lastImage - md1.firstImage + 1 ;
    %Nim = 100 ; %for testing
    
else
    error ('Movie file type can be either TIF or CIN. Aborting.')
end

% centers_xy = zeros(Nim,4);
% centers_yz = zeros(Nim,4);
% centers_xz = zeros(Nim,4);

good = true(Nim,1) ; % keeps the indices where we had all circles in all frames


[centers1, good1, tested_and_good1] = eatOneMovieSkip(md1, Nim, good,  radius1, radius2, skip, 'movie 1 / 3   frame ') ; % xy
[centers2, good2, tested_and_good2] = eatOneMovieSkip(md2, Nim, good1, radius1, radius2, skip, 'movie 2 / 3   frame ') ; % yz
[centers3, good3, tested_and_good3] = eatOneMovieSkip(md3, Nim, good2, radius1, radius2, skip, 'movie 3 / 3   frame ') ; % xz
% 
% [centers1, good1] = eatOneMovie(md1, Nim, good,  radius1, radius2) ; % xy
% [centers2, good2] = eatOneMovie(md2, Nim, good1, radius1, radius2) ; % yz
% [centers3, good3] = eatOneMovie(md3, Nim, good2, radius1, radius2) ; % xz


good = good3;

tested_and_good = tested_and_good1 & tested_and_good2 & tested_and_good3 ;

%
%     if (1)
%         figure(1) ; clf; %#ok<UNRCH>
%         subplot(1,3,2) ; imshow(im1) ; hold on ; viscircles(centersxy(1,:),radiixy(1),'edgecolor','g') ;  viscircles(centersxy(2,:),radiixy(2),'edgecolor','r') ;title('xy')
%         subplot(1,3,3) ; imshow(im2) ; hold on ; viscircles(centersyz(1,:),radiiyz(1),'edgecolor','g') ;  viscircles(centersyz(2,:),radiiyz(2),'edgecolor','r') ; title('yz');
%         subplot(1,3,1) ; imshow(im3) ; hold on ; viscircles(centersxz(1,:),radiixz(1),'edgecolor','g') ;  viscircles(centersxz(2,:),radiixz(2),'edgecolor','r') ; title('xz') ;
%         pause(0.01) ;
%     end

centers1 = centers1(tested_and_good,:) ;
centers2 = centers2(tested_and_good,:) ;
centers3 = centers3(tested_and_good,:) ;


% centers1 = centers1(good,:) ;
% centers2 = centers2(good,:) ;
% centers3 = centers3(good,:) ;

count = size(centers1,1) ;

% arrrange point coordinates in the format that EasyWand likes
% the origin in every image in on the bottom left corner, so all y
% coordinates are transformed to (512-y+1)

% the format of the "points" file is:
% cam1pt1x cam1pt1y cam2pt1x cam2pt1y cam3pt1x cam3pt1y    cam1pt2x cam1pt2y cam2pt2x cam2pt2y cam3pt2x cam3pt2y

M      = zeros(count,12);

M(:,1)  = centers1(1:count,1);
M(:,3)  = centers2(1:count,1);
M(:,5)  = centers3(1:count,1);

M(:,7)  = centers1(1:count,3);
M(:,9)  = centers2(1:count,3);
M(:,11) = centers3(1:count,3);

M(:,2)  = height1 - centers1(1:count,2) + 1;
M(:,4)  = height2 - centers2(1:count,2) + 1;
M(:,6)  = height3 - centers3(1:count,2) + 1;

M(:,8)  = height1 - centers1(1:count,4) + 1;
M(:,10) = height2 - centers2(1:count,4) + 1;
M(:,12) = height3 - centers3(1:count,4) + 1;


% close cin files if needed
if (~isTif)
    myCloseCinFile(md1.cindata) ;
    myCloseCinFile(md2.cindata) ;
    myCloseCinFile(md3.cindata) ;
end


% save data to the same folder as the movies
if (~exist('outputCSVfile', 'var'))
    outputCSVfile = [] ;
end

if (~isempty(outputCSVfile)) && (~strcmp(outputCSVfile(end-3:end),'.csv'))
    csvwrite([outputCSVfile '.csv'],M) ;
else
    csvwrite(outputCSVfile,M) ;
end

return


% ================================================================

function [centers, good, tested_and_good] = eatOneMovieSkip(md, Nim, good, radius1, radius2, skip, displayStr) % works for cine only

sens = 0.95;
metric_threshold = 0.2 ;

tested_and_good = false(Nim,1) ;

filter_flag     = false ;
plot_flag       = false ;

prev_centers = [] ;
prev_radii   = [] ;
prevFound    = false ;
centers      = zeros(Nim,4);

crop_accelerate = false ; % must be false with skip


for i=1:skip:Nim
    
    if (~good(i))
        prevFound = false ;
        continue ;
    end
    
    fprintf('%s %d / %d\n', displayStr, i,Nim) ;
    
    % good(i) is true
    
    % read image
    t1 = md.firstImage + i - 1 ;
    im = myReadCinImage(md.cindata, t1) ;
    
    im = imadjust(im, [0,170]/255, [0,1]) ;
    
    if ~prevFound
        % work on entire image
        if filter_flag
            im = imfilter(im, filt) ;
        end
        [curr_centers, curr_radii, metric] = imfindcircles(im,[radius1, radius2],'ObjectPolarity','dark','sensitivity',sens);
        
        I = metric>=metric_threshold ;
        curr_centers = curr_centers(I,:) ;
        curr_radii   = curr_radii(I,:) ;
        metric       = metric(I) ;
        
        if length(curr_radii)>2 % if more than 2 circles
            [~, ind] = sort(metric,'descend') ;
            curr_centers = curr_centers(ind(1:2),:) ;
            curr_radii   = curr_radii(ind(1:2)) ;
        end
            
        if length(curr_radii)<2 % if less than two circles            
            good(i)     = false ; % if there are not 2 spheres exactly
            prevFound   = false ;
            prev_centers = [] ;
            continue ;
        else %
            % two spheres exactly
            % make sure small sphere is first
            if curr_radii(1)>curr_radii(2)
                curr_centers = [curr_centers(2,:); curr_centers(1,:)];
                curr_radii   = [curr_radii(2),curr_radii(1)]; %#ok<NASGU>
            end
            
            centers(i,:) = [curr_centers(1,:), curr_centers(2,:)] ;
            tested_and_good(i) = true ;
            fprintf('tested_and_good\n')
            
            if crop_accelerate
                prevFound    = true ;
            end
            prev_centers  = curr_centers ;
            prev_radii    = curr_radii ;
        end
        %{
    else % use info from previous frame to crop image around the two spheres and save processing time
        
        
        prev_centers = round(prev_centers) ;
        
        % sphere 1
        % --------
        c0 = prev_centers(1,1) ;
        r0 = prev_centers(1,2) ;
        L  = round(prev_radii(1)*2) ;
        
        r1 = max([1, r0-L]) ;
        r2 = min([md.height, r0+L]) ;
        c1 = max([1, c0-L]) ;
        c2 = min([md.width, c0+L]) ;
        
%         RR = r2 - r1 + 1 ;
%         CC = c2 - c1 + 1 ;
        
        im_crop = im(r1:r2, c1:c2) ;
        
        if filter_flag
            im_crop = imfilter(im_crop, filt) ;
        end
        
        rad_range = round(prev_radii(1) * [0.9, 1.1]) ;
        [cent1, rad1, metric] = imfindcircles(im_crop, rad_range,'ObjectPolarity','dark','sensitivity',sens);
      
        if length(rad1)>1 % if more than 1 circles
            [~, ind] = sort(metric,'descend') ;
            cent1  = cent1(ind(1),:) ;
            rad1   = rad1(ind(1)) ;
        end
        
%         figure(1) ; clf;
%             imshow(im_crop) ; hold on ; viscircles(cent1,rad1,'edgecolor','g') ;
%         title(i) ; pause ;
        
        if numel(rad1)~=1
            good(i)     = false ; % if there are not 2 spheres exactly
            prevFound   = false ;
            prev_centers = [] ;
            continue ;
            figure(1) ; clf;
            imshow(im_crop) ; hold on ; viscircles(cent1,rad1,'edgecolor','g') ;
            
        end
        
        % offset cent1
        cent1 = cent1 + [c1, r1] - 1; 
        %disp('check if offest is correct and not +/- 1') ;
        
        % sphere 2 (only if 1 if found)
        % -----------------------------
        c0 = prev_centers(2,1) ;
        r0 = prev_centers(2,2) ;
        L  = round(prev_radii(2)*2) ;
        
        r1 = max([1, r0-L]) ;
        r2 = min([md.height, r0+L]) ;
        c1 = max([1, c0-L]) ;
        c2 = min([md.width, c0+L]) ;
        
        im_crop = im(r1:r2, c1:c2) ;
        
        if filter_flag
            im_crop = imfilter(im_crop, filt) ;
        end
        
        rad_range = round(prev_radii(2) * [0.9, 1.1]) ;
        [cent2, rad2, metric] = imfindcircles(im_crop,rad_range,'ObjectPolarity','dark','sensitivity',sens);
        
         if length(rad2)>1 % if more than 1 circles
            [~, ind] = sort(metric,'descend') ;
            cent2  = cent2(ind(1),:) ;
            rad2   = rad2(ind(1)) ;
        end
        
        if numel(rad2)~=1
            good(i)     = false ; % if there are not 2 spheres exactly
            prevFound   = false ;
            prev_centers = [] ;
            continue ;
            %figure(1) ; clf;
            %imshow(im_crop) ; hold on ; viscircles(cent2,rad2,'edgecolor','g') ;
            
            
        end
        
        % offset cent1
        cent2 = cent2 + [c1, r1] - 1 ;
        %disp('check if offest is correct and not +/- 1') ;
        
        % two circles are found
        
        if rad1<rad2
            curr_centers = [cent1, cent2] ;
            curr_radii   = [rad1,  rad2 ] ;
        else
            curr_centers = [cent2, cent1] ;
            curr_radii   = [rad2,  rad1 ] ;
        end
        
        if (sum(curr_centers<0)>0)
            keyboard ;
        end
        
        centers(i,:) = curr_centers ;
        tested_and_good(i) = true ;
        
        
        prevFound     = true ;
        prev_centers  = [curr_centers(1:2) ; curr_centers(3:4)] ;
        prev_radii    = curr_radii ;
        
        
        % if did not find using this method, try the standard one (or use
        % while instead of for and make sure we do over this i again with
        % the correct flag)
    %}
    end
    
    if plot_flag
        figure(1) ; clf; imshow(im) ; title(i) ;  hold on ;
        viscircles(centers(i,1:2),curr_radii(1),'edgecolor','g') ;
        viscircles(centers(i,3:4),curr_radii(2),'edgecolor','r') ;
        pause(0.05) ;
    end
    
end

return


%{
% ================================================================

function [centers, good] = eatOneMovie(md, Nim, good, radius1, radius2) % works for cine only

sens = 0.95;

filter_flag     = false ;
plot_flag       = false ;

filt = fspecial('disk',3) ;

prev_centers = [] ;
prev_radii   = [] ;
prevFound    = false ;
centers      = zeros(Nim,4);

crop_accelerate = true ;


for i=1:Nim
    
    fprintf('%d / %d\n',i,Nim) ;
    
    if (~good(i))
        prevFound = false ;
        continue ;
    end
    
    % good(i) is true
    
    % read image
    t1 = md.firstImage + i - 1 ;
    im = myReadCinImage(md.cindata, t1) ;
    
    
    
    if ~prevFound
        % work on entire image
        if filter_flag
            im = imfilter(im, filt) ;
        end
        [curr_centers, curr_radii, metric] = imfindcircles(im,[radius1, radius2],'ObjectPolarity','dark','sensitivity',sens);
        
        if length(curr_radii)>2 % if more than 2 circles
            [~, ind] = sort(metric,'descend') ;
            curr_centers = curr_centers(ind(1:2),:) ;
            curr_radii   = curr_radii(ind(1:2)) ;
        end
            
        if length(curr_radii)<2 % if less than two circles            
            good(i)     = false ; % if there are not 2 spheres exactly
            prevFound   = false ;
            prev_centers = [] ;
            continue ;
        else
            % make sure small sphere is first
            if curr_radii(1)>curr_radii(2)
                curr_centers = [curr_centers(2,:); curr_centers(1,:)];
                curr_radii   = [curr_radii(2),curr_radii(1)]; %#ok<NASGU>
            end
            
            centers(i,:) = [curr_centers(1,:), curr_centers(2,:)] ;
            
            if crop_accelerate
                prevFound    = true ;
            end
            prev_centers  = curr_centers ;
            prev_radii    = curr_radii ;
        end
    else % use info from previous frame to crop image around the two spheres and save processing time
        
        
        prev_centers = round(prev_centers) ;
        
        % sphere 1
        % --------
        c0 = prev_centers(1,1) ;
        r0 = prev_centers(1,2) ;
        L  = round(prev_radii(1)*2) ;
        
        r1 = max([1, r0-L]) ;
        r2 = min([md.height, r0+L]) ;
        c1 = max([1, c0-L]) ;
        c2 = min([md.width, c0+L]) ;
        
%         RR = r2 - r1 + 1 ;
%         CC = c2 - c1 + 1 ;
        
        im_crop = im(r1:r2, c1:c2) ;
        
        if filter_flag
            im_crop = imfilter(im_crop, filt) ;
        end
        
        rad_range = round(prev_radii(1) * [0.9, 1.1]) ;
        [cent1, rad1, metric] = imfindcircles(im_crop, rad_range,'ObjectPolarity','dark','sensitivity',sens);
      
        if length(rad1)>1 % if more than 1 circles
            [~, ind] = sort(metric,'descend') ;
            cent1  = cent1(ind(1),:) ;
            rad1   = rad1(ind(1)) ;
        end
        
%         figure(1) ; clf;
%             imshow(im_crop) ; hold on ; viscircles(cent1,rad1,'edgecolor','g') ;
%         title(i) ; pause ;
        
        if numel(rad1)~=1
            good(i)     = false ; % if there are not 2 spheres exactly
            prevFound   = false ;
            prev_centers = [] ;
            continue ;
            figure(1) ; clf;
            imshow(im_crop) ; hold on ; viscircles(cent1,rad1,'edgecolor','g') ;
            
        end
        
        % offset cent1
        cent1 = cent1 + [c1, r1] - 1; 
        %disp('check if offest is correct and not +/- 1') ;
        
        % sphere 2 (only if 1 if found)
        % -----------------------------
        c0 = prev_centers(2,1) ;
        r0 = prev_centers(2,2) ;
        L  = round(prev_radii(2)*2) ;
        
        r1 = max([1, r0-L]) ;
        r2 = min([md.height, r0+L]) ;
        c1 = max([1, c0-L]) ;
        c2 = min([md.width, c0+L]) ;
        
        im_crop = im(r1:r2, c1:c2) ;
        
        if filter_flag
            im_crop = imfilter(im_crop, filt) ;
        end
        
        rad_range = round(prev_radii(2) * [0.9, 1.1]) ;
        [cent2, rad2, metric] = imfindcircles(im_crop,rad_range,'ObjectPolarity','dark','sensitivity',sens);
        
         if length(rad2)>1 % if more than 1 circles
            [~, ind] = sort(metric,'descend') ;
            cent2  = cent2(ind(1),:) ;
            rad2   = rad2(ind(1)) ;
        end
        
        if numel(rad2)~=1
            good(i)     = false ; % if there are not 2 spheres exactly
            prevFound   = false ;
            prev_centers = [] ;
            continue ;
            figure(1) ; clf;
            imshow(im_crop) ; hold on ; viscircles(cent2,rad2,'edgecolor','g') ;
            
            
        end
        
        % offset cent1
        cent2 = cent2 + [c1, r1] - 1 ;
        %disp('check if offest is correct and not +/- 1') ;
        
        % two circles are found
        
        if rad1<rad2
            curr_centers = [cent1, cent2] ;
            curr_radii   = [rad1,  rad2 ] ;
        else
            curr_centers = [cent2, cent1] ;
            curr_radii   = [rad2,  rad1 ] ;
        end
        
        if (sum(curr_centers<0)>0)
            keyboard ;
        end
        
        centers(i,:) = curr_centers ;
        
        prevFound     = true ;
        prev_centers  = [curr_centers(1:2) ; curr_centers(3:4)] ;
        prev_radii    = curr_radii ;
        
        
        % if did not find using this method, try the standard one (or use
        % while instead of for and make sure we do over this i again with
        % the correct flag)
    end
    
    if plot_flag
        figure(1) ; clf; imshow(im) ; title(i) ;  hold on ;
        viscircles(curr_centers(1:2),curr_radii(1),'edgecolor','g') ;
        viscircles(curr_centers(3:4),curr_radii(2),'edgecolor','r') ;
        pause(0.05) ;
    end
    
end

return
%}