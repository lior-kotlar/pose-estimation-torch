import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import json
from predict_2D_sparse_box import Predictor2D
from predictions_2Dto3D import From2Dto3D
from extract_flight_data import FlightAnalysis
from visualize import Visualizer
import h5py
from utils import get_start_frame
import torch
from models_config import *
from extract_flight_data import create_movie_analysis_h5
import shutil
import csv
from collections import defaultdict
import pickle
from tqdm import tqdm


class Flight3DProcessing:
    WINDOW_SIZE = 31
    CAMERAS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    CAMERA_PAIRS_INDEXES = [0, 1, 2, 3, 4, 5]

    def __init__(self):
        self.check_gpu()

    @staticmethod
    def check_gpu():
        try:
            torch.zeros(4).cuda()
        except:
            print("No GPU found, doesn't use cuda")
        print("TensorFlow version:", tf.__version__, flush=True)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("TensorFlow is using GPU.", flush=True)
        else:
            print("TensorFlow is using CPU.", flush=True)

    @staticmethod
    def create_movie_html(movie_dir_path, name="points_3D_smoothed_ensemble_best.npy"):
        print(movie_dir_path, flush=True)
        start_frame = get_start_frame(movie_dir_path)
        points_path = os.path.join(movie_dir_path, name)
        if os.path.isfile(points_path):
            file_name = 'movie_html.html'
            save_path = os.path.join(movie_dir_path, file_name)
            FA = FlightAnalysis(points_path)
            com = FA.center_of_mass
            x_body = FA.x_body
            y_body = FA.y_body
            points_3D = FA.points_3D
            start_frame = int(start_frame)
            Visualizer.create_movie_plot(com=com, x_body=x_body, y_body=y_body, points_3D=points_3D,
                                         start_frame=start_frame, save_path=save_path)

    @staticmethod
    def predict_3D_points_all_pairs(base_path):
        all_points_file_list = []
        points_3D_file_list = []
        dir_path = os.path.join(base_path)
        dirs = glob.glob(os.path.join(dir_path, "*"))
        for dir in dirs:
            if os.path.isdir(dir):
                all_points_file = os.path.join(dir, "points_3D_all.npy")
                points_3D_file = os.path.join(dir, "points_3D.npy")
                if os.path.isfile(all_points_file):
                    all_points_file_list.append(all_points_file)
                if os.path.isfile(points_3D_file):
                    points_3D_file_list.append(points_3D_file)
        all_points_arrays = [np.load(array_path) for array_path in all_points_file_list]
        big_array_all_points = np.concatenate(all_points_arrays, axis=2)
        return big_array_all_points, all_points_arrays

    @staticmethod
    def find_3D_points_from_ensemble(base_path, test=False):
        result, all_points_list = Flight3DProcessing.predict_3D_points_all_pairs(base_path)
        # todo for debug
        # all_points_list = all_points_list[:2]
        # all_points_list = [all_points_list[i][:, :, :6, :] for i in range(len(all_points_list))]
        #
        final_score, best_points_3D, all_models_combinations, all_frames_scores = Predictor2D.find_3D_points_optimize_neighbors(
            all_points_list)

        print(f"score: {final_score}\n", flush=True)
        if not test:
            smoothed_3D = Predictor2D.smooth_3D_points(best_points_3D)
            Flight3DProcessing.save_points_3D(base_path, [], best_points_3D, smoothed_3D, "best_method")
            save_name = os.path.join(base_path, "all_models_combinations.npy")
            np.save(save_name, all_models_combinations)
        return best_points_3D, smoothed_3D

    @staticmethod
    def save_points_3D(base_path, best_combination, best_points_3D, smoothed_3D, type_chosen):
        score1 = From2Dto3D.get_validation_score(best_points_3D)
        score2 = From2Dto3D.get_validation_score(smoothed_3D)
        From2Dto3D.save_points_3D(base_path, best_points_3D, name=f"points_3D_ensemble_{type_chosen}.npy")
        From2Dto3D.save_points_3D(base_path, smoothed_3D, name=f"points_3D_smoothed_ensemble_{type_chosen}.npy")
        readme_path = os.path.join(base_path, "README_scores_3D_ensemble.txt")
        print(f"score1 is {score1}, score2 is {score2}")
        with open(readme_path, "w") as f:
            f.write(f"The score for the points was {score1}\n")
            f.write(f"The score for the smoothed points was {score2}\n")
            f.write(f"The winning combination was {best_combination}")

    @staticmethod
    def get_cropzone(movie_dir):
        files = os.listdir(movie_dir)
        movie_files = [file for file in files if file.startswith('movie') and file.endswith('.h5')]
        h5_file_name = movie_files[0]
        h5_file_path = os.path.join(movie_dir, h5_file_name)
        cropzone = h5py.File(h5_file_path, 'r')['cropzone'][:]
        return cropzone

    @staticmethod
    def predict_all_movies(base_path, config_path_2D,
                           calibration_path,
                           movies=None,
                           config_functions_inds=None,
                           already_predicted_2D=False,
                           only_create_mp4=False):
        import predictions_2Dto3D
        file_list = []
        dirlist = [base_path]
        if movies is None:
            movies = os.listdir(base_path)
        for sub_dir in movies:
            sub_dir_path = os.path.join(base_path, sub_dir)
            isdir = os.path.isdir(sub_dir_path)
            if isdir:
                dirlist.append(sub_dir_path)
        for dir in dirlist:
            for file in os.listdir(dir):
                if file.startswith('movie') and file.endswith('.h5'):
                    file_path = os.path.join(dir, file)
                    if os.path.isfile(file_path):
                        file_list.append(file_path)
        # file_list = [file_list[-1]]
        config_functions = [
            # config_1_5,
            config_2_5,
            # config_3_5,
            # config_4_5,
            # config_5_5,
            # config_6_5,
            # config_7_5,
            # config_8_5,
            # config_9_5,
            # config_10_5
        ]

        # config_functions = [config_1, config_2,
        #                     config_3, config_4,
        #                     config_5, config_6,
        #                     config_7, config_8]

        if config_functions_inds is not None:
            config_functions = [config_functions[i] for i in config_functions_inds]

        for movie_path in file_list:
            print(movie_path, flush=True)
            dir_path = os.path.dirname(movie_path)

            # UNCOMMENT LATER
            # started_file_path = os.path.join(dir_path, 'started.txt')
            # if not os.path.exists(started_file_path):
            #     with open(started_file_path, 'w') as file:
            #         file.write('Processing started')
            # else:
            #     print(f"Skipping {movie_path}, processing already started.", flush=True)
            #     continue

            done_file_path = os.path.join(dir_path, 'done.txt')

            # UNCOMMENT LATER
            # if os.path.exists(done_file_path):
            #     print(f"Skipping {movie_path}, already processed.")
            #     continue
            if not only_create_mp4:
                Flight3DProcessing.save_predict_code(movie_path)

            with open(config_path_2D) as C:
                config_2D = json.load(C)
                config_2D["box path"] = movie_path
                config_2D["base output path"] = dir_path
                config_func = config_1_5
                config_2D = config_func(config_2D)
                config_2D["calibration data path"] = calibration_path
            new_config_path_save_box = os.path.join(dir_path, 'configuration_predict_2D.json')
            with open(new_config_path_save_box, 'w') as file:
                json.dump(config_2D, file, indent=4)

            predictor = Predictor2D(new_config_path_save_box)
            if not already_predicted_2D:
                if config_functions:
                    predictor.create_base_box()
                    predictor.save_base_box()

                for model in range(len(config_functions)):
                    dir_path = os.path.dirname(movie_path)
                    with open(config_path_2D) as C:
                        config_2D = json.load(C)
                        config_2D["box path"] = movie_path
                        config_2D["base output path"] = dir_path
                        config_func = config_functions[model]
                        config_2D = config_func(config_2D)
                        config_2D["calibration data path"] = calibration_path
                    new_config_path = os.path.join(dir_path, 'configuration_predict_2D.json')
                    with open(new_config_path, 'w') as file:
                        json.dump(config_2D, file, indent=4)
                    try:
                        predictor_model = Predictor2D(new_config_path, load_box_from_sparse=True)
                        predictor_model.run_predict_2D()
                    except Exception as e:
                        print(f"Error while processing movie {movie_path} model {model}: {e}")

            cropzone = predictor.get_cropzone()
            if not only_create_mp4:
                print("started predicting ensemble")
                best_points_3D, smoothed_3D = Flight3DProcessing.find_3D_points_from_ensemble(dir_path)
                reprojected = predictor.triangulate.get_reprojections(best_points_3D, cropzone)
                smoothed_reprojected = predictor.triangulate.get_reprojections(smoothed_3D, cropzone)
                From2Dto3D.save_points_3D(dir_path, reprojected, name="points_ensemble_reprojected.npy")
                From2Dto3D.save_points_3D(dir_path, smoothed_reprojected,
                                          name="points_ensemble_smoothed_reprojected_before_analisys.npy")

            # create analsys
            Flight3DProcessing.create_movie_html(dir_path, name="points_3D_smoothed_ensemble_best_method.npy")
            points_3D_path = os.path.join(dir_path, 'points_3D_smoothed_ensemble_best_method.npy')
            reprojected_points_path = os.path.join(dir_path, 'points_ensemble_smoothed_reprojected.npy')
            box_path = movie_path
            save_path = os.path.join(dir_path, 'movie 2D and 3D.gif')
            movie = os.path.basename(dir_path)
            rotate = True
            try:
                movie_hdf5_path, FA = create_movie_analysis_h5(movie, dir_path, points_3D_path, smooth=True)
                Visualizer.plot_all_body_data(movie_hdf5_path)
                reprojected = predictor.triangulate.get_reprojections(FA.points_3D[FA.first_analysed_frame:], cropzone)
                From2Dto3D.save_points_3D(dir_path, reprojected,
                                          name="points_ensemble_smoothed_reprojected.npy")  # better 2D points
                Visualizer.create_movie_mp4(movie_hdf5_path, save_frames=None, mode='SAVE',
                                            reprojected_points_path=reprojected_points_path,
                                            box_path=box_path, save_path=save_path, rotate=rotate)
            except:
                print("wasn't able to analyes the movie and reproject the points")

            print(f"Finished movie {movie_path}", flush=True)
            with open(done_file_path, 'w') as file:
                file.write('Processing completed.')
        Flight3DProcessing.create_ensemble_results_csv(base_path)

    @staticmethod
    def save_predict_code(movie_path):
        code_dir_path = os.path.join(os.path.dirname(movie_path), "predicting code")
        os.makedirs(code_dir_path, exist_ok=True)
        for file_name in os.listdir('.'):
            if file_name.endswith('.py'):
                full_file_name = os.path.join('.', file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, code_dir_path)
                    print(f"Copied {full_file_name} to {code_dir_path}")

    @staticmethod
    def delete_specific_files(base_path, filenames):
        for root, dirs, files in os.walk(base_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted {file_path}", flush=True)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}", flush=True)

    @staticmethod
    def clean_directories(base_path):
        for root, dirs, files in os.walk(base_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        # Check if the file does not meet the exclusion criteria
                        if not (file.endswith('.mat') or file.startswith('README_mov') or (
                                file.startswith('movie_') and file.endswith('.h5'))):
                            try:
                                os.remove(file_path)
                                print(f"Deleted {file_path}", flush=True)
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}", flush=True)
                    elif os.path.isdir(file_path):
                        # Check if the directory does not meet the exclusion criteria
                        try:
                            shutil.rmtree(file_path)
                            print(f"Deleted directory {file_path}", flush=True)
                        except Exception as e:
                            print(f"Error deleting directory {file_path}: {e}", flush=True)

    @staticmethod
    def predict_and_analyze_directory(base_path,
                                      config_path=r"/cs/labs/tsevi/lior.kotlar/pose-estimation/from 2D to 3D/predict_2D_config.json",
                                      already_predicted_2D=True, calibration_path="",
                                      only_create_mp4=False):
        Flight3DProcessing.predict_all_movies(base_path,
                                              config_path,
                                              calibration_path,
                                              already_predicted_2D=already_predicted_2D,
                                              only_create_mp4=only_create_mp4)

    @staticmethod
    def extract_scores(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            points_score = float(lines[0].split()[-1])
            smoothed_points_score = float(lines[1].split()[-1])
        return points_score, smoothed_points_score

    @staticmethod
    def create_ensemble_results_csv(base_path):
        output_file = os.path.join(base_path, 'ensemble_results.csv')
        data = []

        for dir_name in os.listdir(base_path):
            dir_path = os.path.join(base_path, dir_name)
            if os.path.isdir(dir_path):
                score_file = os.path.join(dir_path, 'README_scores_3D_ensemble.txt')
                if os.path.exists(score_file):
                    points_score, smoothed_points_score = Flight3DProcessing.extract_scores(score_file)
                    data.append({
                        'directory name': dir_name,
                        'points score': points_score,
                        'smoothed points score': smoothed_points_score
                    })

        # Sort data by 'smoothed points score'
        data.sort(key=lambda x: x['smoothed points score'])

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['directory name', 'points score', 'smoothed points score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)


def run_predict_directory():
    # base_path = 'dark 24-1 movies'
    # base_path = 'example datasets'
    # base_path = 'free 24-1 movies'
    # base_path = 'roni movies'
    # Flight3DProcessing.clean_directories(base_path)
    # base_path = 'example datasets'
    cluster = False
    already_predicted_2D = False
    only_create_mp4 = False
    if cluster is True:
        base_path = 'roni bad movies'
        calibration_path = fr"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/{base_path}/calibration file.h5"
        Flight3DProcessing.predict_and_analyze_directory(base_path,
                                                         calibration_path=calibration_path,
                                                         already_predicted_2D=already_predicted_2D,
                                                         only_create_mp4=only_create_mp4)
    else:
        base_path = r"/cs/labs/tsevi/lior.kotlar/pose-estimation/inference_datasets/mov_10/cropped"
        calibration_path = r"/cs/labs/tsevi/lior.kotlar/pose-estimation/inference_datasets/mov_10/calibration file.h5"
        Flight3DProcessing.predict_and_analyze_directory(base_path,
                                                         calibration_path=calibration_path,
                                                         already_predicted_2D=already_predicted_2D,
                                                         only_create_mp4=only_create_mp4)


if __name__ == '__main__':
    # dir_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies\mov14"
    # best_points_3D, smoothed_3D = Flight3DProcessing.find_3D_points_from_ensemble(dir_path)
    # print(dir_path)
    # base_path = r"free 24-1 movies"
    # base_path = "example datasets"
    # Flight3DProcessing.create_ensemble_results_csv(base_path)
    # base_path = "example datasets"
    # filenames = ["started_mp4.txt", "started.txt", "done.txt"]
    # base_path = r"free 24-1 movies"
    # Flight3DProcessing.delete_specific_files(base_path, filenames)
    # base_path = "dark 24-1 movies"
    # Flight3DProcessing.delete_specific_files(base_path, filenames)
    run_predict_directory()
