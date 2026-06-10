function LoadPhantomLibraries()

% loads phcon
if (~libisloaded('phcon'))
    [notfoundPhCon, warningsPhCon] = loadlibrary('phcon.dll','PhConML.h', 'alias', 'phcon')
end
% loads phint
if (~libisloaded('phint'))
    [notfoundPhInt, warningsPhInt] = loadlibrary ('phint.dll','PhIntML.h', 'alias', 'phint')
end
% loads phfile
if (~libisloaded('phfile'))
    [notfoundPhFile, warningsPhFile] = loadlibrary ('phfile.dll','PhFileML.h', 'alias', 'phfile')
end

end

