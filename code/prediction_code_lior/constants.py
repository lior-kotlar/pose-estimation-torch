BODY_POINTS = [7, 15, 16, 17]

# body parts to predict
WINGS = "WINGS"
BODY = "BODY"
WINGS_AND_BODY = "WINGS_AND_BODY"

#model types
ALL_POINTS_ALL_CAMS = "ALL_POINTS_ALL_CAMS" # predicts all points using all cameras at once
PER_WING_ALL_CAMS = "PER_WING_ALL_CAMS" # predicts one wing at a time using all cameras at once
PER_WING_PER_CAM = "PER_WING_PER_CAM" # predicts one wing at a time using one camera at a time
ALL_POINTS_PER_CAM = "ALL_POINTS_PER_CAM" # predicts all points at once using one camera at a time
ALL_POINTS_REPROJECTED_MASKS = "ALL_POINTS_REPROJECTED_MASKS"

# 2d to 3d loading types
CONFIG = "CONFIG"
H5_FILE = "H5_FILE"