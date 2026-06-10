function [HRES,cineStatuses] = PhGetCineStatus(CN)

cs = get(libstruct('tagCINESTATUS'));
for i=1:uint32(PhConConst.MAXCINECNT)
    csa(i) = cs;
end

csp = libpointer('tagCINESTATUS',csa);
[HRES, dummy] = calllib('phcon','PhGetCineStatus',CN,csp);
OutputError(HRES);

for i=1:uint32(PhConConst.MAXCINECNT)
    cineStatuses(i) = get(csp+(i-1),'Value');
end
end