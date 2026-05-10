function new_box = view_masks_perimeter(new_box)
    numFrames = size(new_box, 5);
    numCams = size(new_box, 4);
    for frame=1:numFrames
        for cam=1:numCams
            perim_mask_left = bwperim(new_box(: ,:, 2, cam, frame));
            perim_mask_right = bwperim(new_box(: ,:, 3, cam, frame));
            fly = new_box(: ,:, 1, cam, frame);
            new_box(: ,:, 1, cam, frame) = fly;
            new_box(: ,:, 2, cam, frame) = fly;
            new_box(: ,:, 3, cam, frame) = fly;
            new_box(: ,:, 1, cam, frame) = new_box(: ,:, 1, cam, frame) + perim_mask_left;
            new_box(: ,:, 3, cam, frame) = new_box(: ,:, 3, cam, frame) + perim_mask_right;
        end
    end
end