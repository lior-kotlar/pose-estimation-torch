import shutil
import sys
import os
from datetime import date
import glob
import numpy as np
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
from utils import PREDICTION_CONFIGURATIONS_DIRECTORY, show_interest_points_with_index, PredictConfig, PREDICTION_CODE_DIRECTORY
import torch
from Predictor import Predictor2D
from From_2D_to_3D import From2Dto3D


class PredictingManager:
    def __init__(self, config_path, device):
        self.config = PredictConfig(config_path)
        self.model_config_list = self.config.get_model_config_list()
        self.device = device
        self.movie_path_list = self.configure_movie_list()
        self.base_run_directory = self.create_general_run_directory()
        self.save_prediction_code_and_configurations()

    def configure_movie_list(self):
        movie_list = []
        base_dir_path = self.config.get_input_data_directory()
        dir_list = [base_dir_path]
        directories = os.listdir(base_dir_path)
        for sub_dir in directories:
            sub_dir_path = os.path.join(base_dir_path, sub_dir)
            is_dir = os.path.isdir(sub_dir_path)
            if is_dir:
                dir_list.append(sub_dir_path)
        for dir in dir_list:
            files = os.listdir(dir)
            for file in files:
                if file.startswith('movie') and file.endswith('.h5'):
                    file_path =  os.path.join(dir, file)
                    if os.path.isfile(file_path):
                        movie_list.append(file_path)
        return movie_list

    def create_model_run_directory(self, movie_directory_path,model_type):
        directory_name = os.path.join(movie_directory_path, model_type)
        initial_directory_name = directory_name
        i=1
        while os.path.exists(directory_name):
            directory_name = "%s_%02d" % (initial_directory_name, i)
            i+=1
        os.makedirs(directory_name)
        print(f"Created model run directory at: {directory_name}")
        return directory_name

    def predict_movies(self,
                       only_create_mp4=False):
        for movie_path in self.movie_path_list:
            movie_run_directory_path = os.path.join(self.base_run_directory, os.path.basename(movie_path).replace('.h5',''))
            os.makedirs(movie_run_directory_path, exist_ok=True)
            predictor = None
            for model_config in self.model_config_list:
                
                model_run_directory_path = self.create_model_run_directory(movie_run_directory_path, model_config["model type"])
                self.config.tune_configuration(model_config, movie_path, model_run_directory_path)
                self.config.save_config_as_json(model_run_directory_path)
                predictor = Predictor2D(predict_config=self.config,
                                        device=self.device)
                predictor.create_base_box()
                predictor.save_base_box()
                try:
                    predictor.run_predict_2D()
                except Exception as e:
                    print(f"An error occurred during prediction: {e}")
        
            cropzone = predictor.get_cropzone()
            if not only_create_mp4:
                print("Starting to predict ensemble")
                best_points_3D, smoothed_3D = find_3D_points_from_ensemble(movie_run_directory_path)
                reprojected = predictor.triangulator.get_reprojections(best_points_3D, cropzone)
                smoothed_reprojected = predictor.triangulator.get_reprojections(smoothed_3D, cropzone)
                From2Dto3D.save_points_3D(movie_run_directory_path, reprojected, name="points_ensemble_reprojected.npy")
                From2Dto3D.save_points_3D(movie_run_directory_path, smoothed_reprojected,
                                          name="points_ensemble_smoothed_reprojected_before_analisys.npy")

    def create_general_run_directory(self):
        experiment_name = os.path.basename(self.config.get_input_data_directory())
        general_run_name = experiment_name
        base_output_directory = self.config.get_output_directory()
        run_path = os.path.join(base_output_directory, general_run_name)
        os.makedirs(run_path, exist_ok=True)
        print(f"Created run directory at: {run_path}")
        return run_path
        

    def save_prediction_code_and_configurations(self):
        code_dir_path = os.path.join(self.base_run_directory, "predicting code")
        config_dir_path = os.path.join(self.base_run_directory, "predicting configurations")
        os.makedirs(code_dir_path, exist_ok=True)
        os.makedirs(config_dir_path, exist_ok=True)
        for file_name in os.listdir(PREDICTION_CODE_DIRECTORY):
            if file_name.endswith('.py'):
                full_file_path = os.path.join(PREDICTION_CODE_DIRECTORY, file_name)
                if os.path.isfile(full_file_path):
                    shutil.copy(full_file_path, code_dir_path)
                    file_name_only = os.path.basename(full_file_path)
                    print(f"Copied {full_file_path} to {code_dir_path}")
        for file_name in os.listdir(PREDICTION_CONFIGURATIONS_DIRECTORY):
            if file_name.endswith('.json'):
                full_file_path = os.path.join(PREDICTION_CONFIGURATIONS_DIRECTORY, file_name)
                if os.path.isfile(full_file_path):
                    shutil.copy(full_file_path, config_dir_path)
                    file_name_only = os.path.basename(full_file_path)
                    print(f"Copied {full_file_path} to {config_dir_path}")


def predict_sample_save(model, sample, save_directory, label=None):
    """
    Predict the output for a single sample using the provided model.

    Parameters:
    model: The trained model used for prediction.
    sample: The input sample for which the prediction is to be made.

    Returns:
    The predicted output for the input sample.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Convert the sample to a tensor if it's not already
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        
        # Add batch dimension if necessary
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        
        # Perform the prediction
        prediction = model(sample)

        show_interest_points_with_index(
                                        sample=sample,
                                        label=prediction,
                                        save_directory=save_directory,
                                        filename="predicted_sample.png"
                                        )

        show_interest_points_with_index(
                                        sample=sample,
                                        label=label,
                                        save_directory=save_directory,
                                        filename="ground_truth.png"
                                        )

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

def find_3D_points_from_ensemble(base_path, test=False):
        result, all_points_list = predict_3D_points_all_pairs(base_path)
        final_score, best_points_3D, all_models_combinations, all_frames_scores = Predictor2D.find_3D_points_optimize_neighbors(
            all_points_list)

        print(f"score: {final_score}\n", flush=True)
        if not test:
            smoothed_3D = Predictor2D.smooth_3D_points(best_points_3D)
            save_points_3D(base_path, [], best_points_3D, smoothed_3D, "best_method")
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

def main():
    config_path = sys.argv[1]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for prediction")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction")
    predictor = PredictingManager(config_path, device)
    predictor.predict_movies()
    
    
if __name__ == "__main__":
    main()
