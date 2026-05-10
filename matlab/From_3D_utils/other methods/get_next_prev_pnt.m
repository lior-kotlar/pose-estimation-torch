function [next_pnt, prev_pnt] = get_next_prev_pnt(pnt_num, modulo)
    next_pnt = mod(pnt_num + 1, modulo);
    prev_pnt = mod(pnt_num - 1, modulo);
    if prev_pnt == 0 prev_pnt = modulo; end 
    if next_pnt == 0 next_pnt = modulo; end
end