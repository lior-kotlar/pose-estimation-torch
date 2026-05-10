function new_box = reshape_box(box, masks_view)
    % reshape to (192   192     3     4   520)
    max_val = max(box, [], 'all');
    if max_val > 1
        box = box/255;
    end
    if size(box, 1) == 20
        box = permute(box, [2, 3, 1, 4]);
    end
    if masks_view && size(box, 3) == 20
        box = box(:, :, [2,4,5 ,7,9,10, 12,14,15 ,17,19,20], :);
    end
    if size(box, 1) == 5
        box = permute(box, [2, 3, 1, 4]);
    end
    if size(box, 3) == 5
        box = box(:, :, [2,4,5], :);
    end
    if  ~masks_view 
        box = box(:, :, [1,2,3 ,6,7,8, 11,12,13 ,16,17,18], :);  
    end
    numFrames = size(box, 4);
    numCams = 4;
    new_box = nan(192, 192, 3, numCams, numFrames);
    if size(box, 3) == 12
        for frame=1:numFrames
            new_box(: ,:, :, 1, frame) = box(:,:, (1:3), frame);
            new_box(: ,:, :, 2, frame) = box(:,:, (4:6), frame);
            new_box(: ,:, :, 3, frame) = box(:,:, (7:9), frame);
            new_box(: ,:, :, 4, frame) = box(:,:, (10:12), frame);
        end
    end
    
    if size(box, 3) == 3
        numFrames = int64(size(box, 4)/4);
        new_box = nan(192, 192, 3, numCams, numFrames);
        new_box(: ,:, :, 1, :) = box(:,:, :, 1:numFrames);
        new_box(: ,:, :, 2, :) = box(:,:, :, (numFrames + 1): numFrames*2);
        new_box(: ,:, :, 3, :) = box(:,:, :, (numFrames*2 + 1):(numFrames*3));
        new_box(: ,:, :, 4, :) = box(:,:, :, (numFrames*3 + 1):numFrames*4);
    end
end