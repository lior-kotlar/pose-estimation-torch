classdef PoolRefresher < handle
%Class to manage the online list of cameras
    
    properties(Access = private)
        CameraList = [];
    end
    
    methods(Access = private, Static)
        function cameraList = GetEthernetCameras()
            [HRES, camCount] = PhGetCameraCount();
            %cameraList = {};
            if (camCount>0)
                for CamNr = 1:camCount
                    cam = Camera(CamNr-1);
                    cameraList(CamNr) = cam;
                end
            else
                %place a default simulated camera in the case of no camera connected
                PhAddSimulatedCamera(660, 1234);
                simCam = Camera(0);
                cameraList(1) = simCam;
            end
        end
    end
    
    methods
        function poolRefresher = PoolRefresher()
            poolRefresher.CameraList = PoolRefresher.GetEthernetCameras();
        end
        
        function cam = GetCameraAt(this, camIndex)
            if (camIndex>=1 && camIndex<=length(this.CameraList))
                cam = this.CameraList(camIndex);
            else
                cam = [];
            end
        end
        
        function camListLength = GetCameraListLength(this)
            camListLength = length(this.CameraList);
        end

        function RefreshCameras(this)
            %Checks if the current camera list is up to date. If not refreshes the list.
            poolChanged = PhCheckCameraPool();
            if (poolChanged)
                PhNotifyDeviceChangeCB();
                %get the new list of cameras
                this.CameraList = this.GetEthernetCameras();
            end
        end
        
        function camListIndex = GetCameraIndexForSerial(this, serial)
            %If the camera is found, return the camera list index, else return -1
            count = this.GetCameraListLength();
            for i=1:count
                cam = this.CameraList(i);
                if (serial == cam.Serial)
                    camListIndex = i;
                    return;
                end
            end
            %no camera found
            camListIndex = -1;
        end
    end
    
end

