def config_1(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_2(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = r"models/3 good cameras/MODEL_18_POINTS_3_GOOD_CAMERAS_Jan 03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_3(config):
    # 3 good cameras 2
    config["wings pose estimation model path"] = r"models/3 good cameras/MODEL_18_POINTS_3_GOOD_CAMERAS_Jan 03_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_4(config):
    # 2 passes reprojected masks
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = "models/per wing/MODEL_18_POINTS_PER_WING_Jan 20/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 0
    return config


def config_5(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/4 cameras/concatenated encoder/ALL_CAMS_18_POINTS_Jan 19_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 0
    return config


def config_6(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/4 cameras/concatenated encoder/ALL_CAMS_18_POINTS_Jan 20_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 0
    return config


def config_7(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/per wing/different seed tough augmentations/MODEL_18_POINTS_PER_WING_Apr 07_08/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 0
    return config


def config_8(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/per wing/different seed tough augmentations/MODEL_18_POINTS_PER_WING_Apr 07_09/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 0
    return config


# new configurations
def config_1_new(config):
    #
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May 29/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_2_new(config):
    #
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined 3/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_3_new(config):
    # 3 good cameras 2
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_3_GOOD_CAMERAS_May 29/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_4_new(config):
    # 2 passes reprojected masks
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["wings pose estimation model path second path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined 3/best_model.h5"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    return config


def config_5_new(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 3.0/ALL_CAMS_18_POINTS_May 29/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again 3D consistent"] = 1
    return config


def config_6_new(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined 3/best_model.h5"
    config[
        "wings pose estimation model path second path"] = r"models 3.0/ALL_CAMS_ALL_POINTS_May 29/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_ALL_POINTS"
    config["predict again 3D consistent"] = 1
    return config


def config_7_new(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined 3/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 3.0/MODEL_18_POINTS_3_GOOD_CAMERAS_May 29/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    return config


def config_8_new(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models 3.0/MODEL_18_POINTS_3_GOOD_CAMERAS_May 29/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 3.0/MODEL_18_POINTS_PER_WING_May on combined 3/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    return config


# new configurations
def config_1_4(config):
    #
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_PER_WING_Jun 01_03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_2_4(config):
    #
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_PER_WING_Jun 01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_3_4(config):
    # 3 good cameras 2
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_3_GOOD_CAMERAS_Jun 01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_4_4(config):
    # 2 passes reprojected masks
    config["wings pose estimation model path"] = r"models 4.0/ALL_POINTS_MODEL_Jun 01/best_model.h5"
    config["model type"] = "ALL_POINTS"
    config["predict again 3D consistent"] = 0
    return config


def config_5_4(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_PER_WING_Jun 01/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 4.0/ALL_CAMS_18_POINTS_Jun 01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again 3D consistent"] = 1
    return config


def config_6_4(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_PER_WING_Jun 01/best_model.h5"
    config[
        "wings pose estimation model path second path"] = r"models 4.0/ALL_CAMS_ALL_POINTS_Jun 01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_ALL_POINTS"
    config["predict again 3D consistent"] = 1
    return config


def config_7_4(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_PER_WING_Jun 01_05/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 4.0/MODEL_18_POINTS_3_GOOD_CAMERAS_Jun 01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    return config


def config_8_4(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models 4.0/MODEL_18_POINTS_3_GOOD_CAMERAS_Jun 01/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 4.0/MODEL_18_POINTS_PER_WING_Jun 01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    return config


# models 5.0
models_path = '/cs/labs/tsevi/lior.kotlar/pose-estimation/from 2D to 3D/models 5.0/'
def config_1_5(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = "/cs/labs/tsevi/lior.kotlar/pose-estimation/models/MODEL_18_POINTS_PER_WING_Apr 28_01/best_model.keras"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_2_5(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = f"/cs/labs/tsevi/lior.kotlar/pose-estimation/models/MODEL_18_POINTS_PER_WING_Jul 07/best_model.keras"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_3_5(config):
    # 3 good cameras 2
    config["wings pose estimation model path"] = f"/cs/labs/tsevi/lior.kotlar/pose-estimation/models/MODEL_18_POINTS_3_GOOD_CAMERAS_Jul 09/weights/weights.050-0.000085914.keras"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 0
    return config


def config_4_5(config):
    # 2 passes reprojected masks
    config["wings pose estimation model path"] = f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 09 not reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/per wing/MODEL_18_POINTS_PER_WING_Jun 09_02 0.7-1.3 reprojected/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config


def config_5_5(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] =f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 09 0.7-1.3 not reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/4 cameras/ALL_CAMS_18_POINTS_Jun 09/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config


def config_6_5(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 09 0.7-1.3 not reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/4 cameras/ALL_CAMS_18_POINTS_Jun 11_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config


def config_7_5(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 09 0.7-1.3 not reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/per wing/MODEL_18_POINTS_PER_WING_Jun 09_04 0.8-1.2 reprojected/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config


def config_8_5(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 09 0.7-1.3 not reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/per wing/MODEL_18_POINTS_PER_WING_Jun 09_02 0.7-1.3 reprojected/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config


def config_9_5(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 16_01_not_reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/4 cameras/ALL_CAMS_ALL_POINTS_Jun 16/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_ALL_POINTS"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config


def config_10_5(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = f"{models_path}per wing/MODEL_18_POINTS_PER_WING_Jun 16_01_not_reprojected/best_model.h5"
    config["wings pose estimation model path second path"] = r"models 5.0/per wing/MODEL_18_POINTS_PER_WING_Jun 16_01_not_reprojected/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again 3D consistent"] = 1
    config["use reprojected masks"] = 1
    return config
