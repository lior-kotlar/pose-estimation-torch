classdef(Sealed) CameraStatus
    
    properties(Constant)
        NotAvailable = 1;
        Preview = 2;
        RecWaitingForTrigger = 3;
        RecPostriggerFrames = 4;
    end
    
    methods (Access = private)
        function out = CameraStatus()
        end
    end
    
end

