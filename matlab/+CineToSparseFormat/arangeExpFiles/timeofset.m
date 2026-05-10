function [tm] = timeofset(tm,hourofset,minofset,secofset,milisecofset)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
a = tm;

tm(4) = mod((tm(4) + milisecofset),1000);
tm(3) = mod((tm(3) + secofset),60) + floor((a(4)+ milisecofset)/1000);
tm(2) = mod((tm(2) + minofset),60)+ floor((a(3) + secofset)/60);
tm(1) =  mod((tm(1) + hourofset),24)+ floor((a(2) + minofset)/60);

end