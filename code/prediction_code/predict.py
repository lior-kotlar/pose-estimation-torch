import sys
import os
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
from utils import show_interest_points_with_index, readfile, Predict_config
import training_code.Network as Network
import torch

class GeneralPredictor:
    def __init__(self, config_path):
        self.config = Predict_config(config_path)
        self.model_config_list = self.config.get_model_config_list()

    def predict_movie(self):
        pass

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

    pass
    
    
if __name__ == "__main__":
    main()
