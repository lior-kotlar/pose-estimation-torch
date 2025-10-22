BODY_POINTS = [7, 15, 16, 17]

# body parts to predict
WINGS = "WINGS"
BODY = "BODY"
WINGS_AND_BODY = "WINGS_AND_BODY"

#model types
ALL_CAMS_ALL_POINTS = "ALL_CAMS_ALL_POINTS" # predict all points (wings and body) using all cameras at once
ALL_CAMS_PER_WING = "ALL_CAMS_PER_WING" # predict all points (wings and body) using all cameras for each wing separately