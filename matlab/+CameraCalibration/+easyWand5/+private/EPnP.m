function [R,t] = EPnP(XX,xx)

[R,t]= CameraCalibration.easyWand5.private.efficient_pnp(XX.',xx.',diag([1 1 1]));

return