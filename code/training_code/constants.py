
# MODEL TYPES
HEAD_TAIL_PER_CAM = "HEAD_TAIL_PER_CAM"
HEAD_TAIL_ALL_CAMS = "HEAD_TAIL_ALL_CAMS"
HEAD_TAIL_PER_CAM_POINTS_LOSS = "HEAD_TAIL_PER_CAM_POINTS_LOSS"
MODEL_PER_CAM_PER_WING = "MODEL_PER_CAM_PER_WING" # predicts one wing at a time, one camera at a time, including body points
MODEL_PER_CAM_PER_WING_PICK_3_BEST_CAMERAS = "MODEL_PER_CAM_PER_WING_3_GOOD_CAMERAS" # choose best 3 cameras out of 4
MODEL_PER_CAM_PER_WING_3_CAMERAS_ONLY = "MODEL_PER_CAM_PER_WING_3_CAMERAS_ONLY" #there are only 3 cameras to begin with

# general model types. only used for arguments in function like split_per_wing.
# a model can't have these types, only the specific ones above.
PER_WING_MODEL = 'PER_WING_MODEL'
ALL_POINTS_MODEL = 'ALL_POINTS_MODEL'

# SET TYPES
MOVIE_TRAIN_SET = "MOVIE_TRAIN_SET"
RANDOM_TRAIN_SET = "RANDOM_TRAIN_SET"

IMAGE_SIZE = 192
model_types_to_channels = {
    MODEL_PER_CAM_PER_WING: 4,
    MODEL_PER_CAM_PER_WING_PICK_3_BEST_CAMERAS: 4,
    MODEL_PER_CAM_PER_WING_3_CAMERAS_ONLY: 4
}