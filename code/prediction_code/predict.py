import sys
import os
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
from utils import show_interest_points_with_index, readfile, Config
import training_code.Network as Network
import torch

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

    predicting_data_path = "/cs/labs/tsevi/lior.kotlar/pose-estimation/training_datasets/combined_dataset.h5"
    training_configuration_path = sys.argv[1]
    config = Config(training_configuration_path)
    base_directory = config.get_base_output_directory()
    model_path = os.path.join(base_directory, "weights/best_model.pth")
    save_predictions_directory = os.path.join(base_directory,"viz_pred")
    if not os.path.exists(save_predictions_directory):
        exit("Error: Directory {} does not exist".format(save_predictions_directory))
    
    
if __name__ == "__main__":
    main()
