function dist = get_distance_from_mask_to_point(mask, point)
    D = bwdist(mask);
    dist = D(point(2), point(1));
end