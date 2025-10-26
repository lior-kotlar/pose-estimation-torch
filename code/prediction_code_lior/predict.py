import shutil
import sys
import os
from datetime import date
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
from utils import PREDICTION_CONFIGURATIONS_DIRECTORY, show_interest_points_with_index, Predict_config, PREDICTION_CODE_DIRECTORY
import torch
from Predictor import Predictor2D



class PredictingManager:
    def __init__(self, config_path, device):
        self.config = Predict_config(config_path)
        self.model_config_list = self.config.get_model_config_list()
        self.device = device
        self.base_run_directory = self.create_run_directory()
        self.save_prediction_code_and_configurations()

    def configure_movie_list(self):
        movie_list = []
        dir_list = [self.config.get_input_data_directory()]
        

    def predict_movie(self):
        for model_config in self.model_config_list:
            self.config.tune_configuration(model_config)
            predictor = Predictor2D(self.config, self.device)

    def create_run_directory(self):
        movie_name = os.path.basename(self.config.input_data_directory).split(".")[0]
        run_name = f"{movie_name}_{date.today().strftime('%b_%d')}"
        base_output_directory = self.config.get_output_directory()
        run_path = os.path.join(base_output_directory, run_name)
        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1
        os.makedirs(run_path)
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

def main():
    config_path = sys.argv[1]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for prediction")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction")
    predictor = PredictingManager(config_path, device)
    
    
    
if __name__ == "__main__":
    main()
