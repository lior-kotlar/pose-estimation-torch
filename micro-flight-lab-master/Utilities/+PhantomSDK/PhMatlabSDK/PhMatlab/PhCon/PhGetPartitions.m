function [HRES,rCount,rPartitionSizes] = PhGetPartitions(CN)

pCount=libpointer('uint32Ptr',uint32(PhConConst.MAXCINECNT));
psz=zeros(uint32(PhConConst.MAXCINECNT),1,'uint32');
pPartitionSize=libpointer('uint32Ptr',psz);

[HRES, rCount, rp] = calllib('phcon','PhGetPartitions',CN, pCount, pPartitionSize);
OutputError(HRES);

rPartitionSizes = zeros(rCount,1,'uint32');
for i=1:rCount
    rPartitionSizes(i) = rp(i);
end

end

