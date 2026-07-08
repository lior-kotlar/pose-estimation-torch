import sys
import os
import time
import json
from datetime import date
import glob
import numpy as np
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
from utils import get_start_frame, show_interest_points_with_index, PredictConfig
from pipeline_timing import record as record_timing, earliest_start
from plot_wing_angles import plot_one as plot_wing_angles_one
import h5py
import torch
from Predictor import Predictor2D
from From_2D_to_3D import From2Dto3D
from extract_flight_data import FlightAnalysis
from Visualizer import Visualizer
from Triangulator import Triangulator
from extract_flight_data import create_movie_analysis_h5


class PredictingManager:
    def __init__(self, config_path, device):
        self.config = PredictConfig(config_path)
        self.model_config_list = self.config.get_model_config_list()
        self.device = device
        self.movie_path_list = self.configure_movie_list()
        self.base_run_directory = self.create_general_run_directory()

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
                if file.startswith('mov') and file.endswith('.h5'):
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
        timings_path = self.config.get_pipeline_timings_path()
        for movie_path in self.movie_path_list:
            movie_run_directory_path = os.path.join(self.base_run_directory, os.path.basename(movie_path).replace('.h5',''))
            os.makedirs(movie_run_directory_path, exist_ok=True)
            # Capture post-intersection frame count so timing rows include
            # the normalizer for elapsed-per-frame analysis.
            n_movie_frames = None
            try:
                with h5py.File(movie_path, "r") as _f:
                    n_movie_frames = int(_f["cropzone"].shape[0])
            except Exception:
                pass
            t_movie_start = time.time()
            predictor = None
            for model_config in self.model_config_list:
                # Name the per-member output dir by the model's registry name
                # (e.g. all_cams_dil3) when available, else fall back to the
                # model type. Keeps outputs and the selection report readable.
                model_dir_label = model_config.get("name", model_config["model type"])
                model_run_directory_path = self.create_model_run_directory(movie_run_directory_path, model_dir_label)
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
        
            cropzone = Predictor2D.get_cropzone(movie_path)
            if predictor is None:
                calibration_data_path, image_height, image_width = self.config.get_triangulator_data()
                triangulator = Triangulator(calibration_data_path, image_height, image_width)
            else:
                triangulator = predictor.triangulator
            if not only_create_mp4:
                print("Starting to predict ensemble")
                best_points_3D, smoothed_3D = find_3D_points_from_ensemble(
                    movie_run_directory_path, max_models=self.config.get_max_ensemble_models())
                reprojected = triangulator.get_reprojections(best_points_3D, cropzone)
                smoothed_reprojected = triangulator.get_reprojections(smoothed_3D, cropzone)
                From2Dto3D.save_points_3D(movie_run_directory_path, reprojected, name="points_ensemble_reprojected.npy")
                From2Dto3D.save_points_3D(movie_run_directory_path, smoothed_reprojected,
                                          name="points_ensemble_smoothed_reprojected_before_analysis.npy")
                
            self.create_movie_html(movie_run_directory_path, name="points_3D_smoothed_ensemble_best_method.npy")
            points_3D_path = os.path.join(movie_run_directory_path, 'points_3D_smoothed_ensemble_best_method.npy')
            reprojected_points_path = os.path.join(movie_run_directory_path, 'points_ensemble_smoothed_reprojected.npy')
            box_path = movie_path
            save_path = os.path.join(movie_run_directory_path, 'movie 2D and 3D.mp4')
            movie = os.path.basename(movie_run_directory_path)
            rotate = True
            try:
                movie_hdf5_path, FA = create_movie_analysis_h5(movie, movie_run_directory_path, points_3D_path, smooth=True)
                Visualizer.plot_all_body_data(movie_hdf5_path)
                reprojected = triangulator.get_reprojections(FA.points_3D[FA.first_analysed_frame:], cropzone)
                From2Dto3D.save_points_3D(movie_run_directory_path, reprojected,
                                          name="points_ensemble_smoothed_reprojected.npy")  # better 2D points
                Visualizer.create_movie_mp4(movie_hdf5_path, save_frames=None, mode='SAVE',
                                            reprojected_points_path=reprojected_points_path,
                                            box_path=box_path, save_path=save_path, rotate=rotate)
            except Exception as e:
                print(f"wasn't able to analyes the movie and reproject the points: {e}")
                exit(1)

            t_predict_end = time.time()
            # Record under "mov<N>" (matches process_experiment.py's rows so
            # they join on the same key). Falls back to the h5 basename
            # when the leading "mov_<N>_" pattern is missing.
            h5_base = os.path.basename(movie_path).replace('.h5', '')
            parts = h5_base.split('_')
            movie_label = f"mov{parts[1]}" if len(parts) > 1 and parts[0] == "mov" and parts[1].isdigit() else h5_base
            record_timing(timings_path, movie_label, "predict",
                          t_movie_start, t_predict_end,
                          n_frames=n_movie_frames)
            # Plot wing angles from the analysis h5 just written. Plot
            # failure does NOT abort the pipeline.
            t_step_end = t_predict_end
            try:
                plot_wing_angles_one(movie_hdf5_path, units="ms")
                t_step_end = time.time()
                record_timing(timings_path, movie_label, "plot",
                              t_predict_end, t_step_end)
            except Exception as e:
                print(f"wing-angles plot failed: {e}", flush=True)
            # End-to-end "total" row: from the earliest step recorded for
            # this movie (in the prep job's CSV rows) through end of plot.
            # Falls back to t_movie_start if no prep rows exist (predict-only
            # runs), in which case "total" equals "predict" + "plot".
            earliest = earliest_start(timings_path, movie_label) or t_movie_start
            record_timing(timings_path, movie_label, "total",
                          earliest, t_step_end,
                          n_frames=n_movie_frames)
            print(f"Finished movie {movie_path}", flush=True)
                
            
    def create_movie_html(self, movie_dir_path, name):
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

    def create_general_run_directory(self):
        # Prefer an explicit "general run name" from the config (set by
        # predict_array.sh so every task in a SLURM array lands in one
        # parent). Fall back to the data-directory basename for direct
        # single-invocation runs on an experiment dir.
        run_name = self.config.get_general_run_name() \
            or os.path.basename(self.config.get_input_data_directory().rstrip(os.sep))
        run_path = os.path.join(self.config.get_output_directory(), run_name)
        os.makedirs(run_path, exist_ok=True)
        print(f"Created run directory at: {run_path}")
        return run_path


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

def resolve_member_name(member_dir):
    """Human-readable name for an ensemble member's run directory.

    The subdir itself is named by model type (e.g. PER_WING_ALL_CAMS_01), which
    doesn't reveal which trained model it was. Each member also saved a config
    holding the pose-estimation model path, so append the trained variant (the
    parent dir of best_model.pt) when available: e.g.
    "PER_WING_ALL_CAMS_01 (ALL_CAMS_PER_WING_DIL3_Jul 05_01)"."""
    subdir = os.path.basename(member_dir.rstrip(os.sep))
    for cfg_name in ("specific_configuration.json", "configuration.json"):
        cfg_path = os.path.join(member_dir, cfg_name)
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    model_path = json.load(f).get("wings pose estimation model path")
                if model_path:
                    variant = os.path.basename(os.path.dirname(model_path))
                    return f"{subdir} ({variant})"
            except Exception:
                pass
    return subdir


def list_model_names(base_path):
    """Model directories that contributed an ensemble member, in the same
    deterministic (sorted) order used to build the ensemble. The position in
    this list is the model index used throughout the selection bookkeeping
    (``all_models_combinations`` axis 2, ``model_combination`` entries)."""
    names = []
    for dir in sorted(glob.glob(os.path.join(base_path, "*"))):
        if os.path.isdir(dir) and os.path.isfile(os.path.join(dir, "points_3D_all.npy")):
            names.append(resolve_member_name(dir))
    return names


def predict_3D_points_all_pairs(base_path):
        all_points_file_list = []
        points_3D_file_list = []
        model_names = []
        dir_path = os.path.join(base_path)
        # Sort so the model order is deterministic (glob order is arbitrary).
        # model_names stays aligned with all_points_arrays / the model index.
        dirs = sorted(glob.glob(os.path.join(dir_path, "*")))
        for dir in dirs:
            if os.path.isdir(dir):
                all_points_file = os.path.join(dir, "points_3D_all.npy")
                points_3D_file = os.path.join(dir, "points_3D.npy")
                if os.path.isfile(all_points_file):
                    all_points_file_list.append(all_points_file)
                    model_names.append(resolve_member_name(dir))
                if os.path.isfile(points_3D_file):
                    points_3D_file_list.append(points_3D_file)
        all_points_arrays = [np.load(array_path) for array_path in all_points_file_list]
        big_array_all_points = np.concatenate(all_points_arrays, axis=2)
        return big_array_all_points, all_points_arrays, model_names

def find_3D_points_from_ensemble(base_path, test=False, max_models=None):
    points_save_path = os.path.join(base_path, "points_3D_ensemble_best_method.npy")
    smoothed_save_path = os.path.join(base_path, "points_3D_smoothed_ensemble_best_method.npy")
    combinations_save_path = os.path.join(base_path, "all_models_combinations.npy")
    if (os.path.exists(points_save_path) and 
        os.path.exists(smoothed_save_path) and 
        os.path.exists(combinations_save_path)):
        
        print(f"Found existing output files in {base_path}. Loading from disk and skipping optimization...", flush=True)

        # Load the arrays
        best_points_3D = np.load(points_save_path)
        smoothed_3D = np.load(smoothed_save_path)
        if not test:
            # Optimization is cached, but the human-readable model-selection
            # report is cheap to (re)build from the saved combinations array.
            try:
                all_models_combinations = np.load(combinations_save_path)
                save_model_selection_report(base_path, all_models_combinations,
                                            list_model_names(base_path))
            except Exception as e:
                print(f"could not (re)generate model-selection report from cache: {e}", flush=True)
        return best_points_3D, smoothed_3D

    result, all_points_list, model_names = predict_3D_points_all_pairs(base_path)
    final_score, best_points_3D, all_models_combinations, all_frames_scores = Predictor2D.find_3D_points_optimize_neighbors(
        all_points_list, max_models=max_models)

    print(f"score: {final_score}\n", flush=True)
    if not test:
        smoothed_3D = Predictor2D.smooth_3D_points(best_points_3D)
        save_points_3D(base_path, [], best_points_3D, smoothed_3D, "best_method")
        save_name = os.path.join(base_path, "all_models_combinations.npy")
        np.save(save_name, all_models_combinations)
        save_model_selection_report(base_path, all_models_combinations, model_names,
                                    all_frames_scores=all_frames_scores)
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


# joint-group order of the leading axis of all_models_combinations, matching
# find_3D_points_optimize_neighbors in Predictor.py.
ENSEMBLE_GROUP_NAMES = ['left_wing', 'right_wing', 'head_tail', 'side_points']


def _strip_frames_scores(all_frames_scores):
    """Turn the raw per-frame score bookkeeping into a compact JSON-friendly
    structure: per joint-group, per frame, the score of every model-subset that
    was tried. The (redundant) per-combination 3D points are dropped."""
    out = {}
    for group_name, group in zip(ENSEMBLE_GROUP_NAMES, all_frames_scores):
        group_out = []
        for frame_scores in group:
            group_out.append([
                {'model_combination': [int(i) for i in d['model_combination']],
                 'score': float(d['score'])}
                for d in frame_scores
            ])
        out[group_name] = group_out
    return out


def save_model_selection_report(base_path, all_models_combinations, model_names,
                                all_frames_scores=None):
    """Summarise how often each ensemble member was chosen by the selector.

    all_models_combinations has shape (num_groups, num_frames, num_models,
    num_camera_pairs); a 1 marks a (model, camera-pair) that was part of the
    winning subset for that frame/joint-group. Writes:
      - model_index_legend.json         model index -> model directory name
      - ensemble_model_selection_summary.json / .txt  per-model usage stats
      - model_selection_visualizations/  per-group plots
      - all_frames_scores.json          (optional) full score margins
    """
    all_models_combinations = np.asarray(all_models_combinations)
    num_groups, num_frames, num_models, num_candidates = all_models_combinations.shape

    # Fall back to positional names if the directory listing didn't line up.
    if not model_names or len(model_names) != num_models:
        model_names = [f"model_{m}" for m in range(num_models)]

    legend = {int(m): model_names[m] for m in range(num_models)}
    with open(os.path.join(base_path, "model_index_legend.json"), "w") as f:
        json.dump(legend, f, indent=4)

    summary = {
        "model_index_legend": legend,
        "num_frames": int(num_frames),
        "num_camera_pairs": int(num_candidates),
        "per_group": {},
        "overall": {},
    }

    # frames_selected[g, m] = number of frames where model m was in the winning
    # subset for joint-group g (any camera-pair). usage counts model x cam-pair.
    frames_selected = np.zeros((num_groups, num_models), dtype=int)
    for g in range(num_groups):
        models_g = all_models_combinations[g]                    # (frames, models, pairs)
        selected = np.any(models_g > 0, axis=2)                  # (frames, models)
        frames_selected[g] = selected.sum(axis=0)
        usage = models_g.sum(axis=(0, 2))                        # model x cam-pair count
        summary["per_group"][ENSEMBLE_GROUP_NAMES[g]] = {
            model_names[m]: {
                "fraction_of_frames_selected": float(frames_selected[g, m] / num_frames)
                if num_frames else 0.0,
                "model_x_camerapair_selections": int(usage[m]),
            }
            for m in range(num_models)
        }

    total_slots = num_groups * num_frames
    for m in range(num_models):
        summary["overall"][model_names[m]] = {
            "fraction_of_frames_selected": float(frames_selected[:, m].sum() / total_slots)
            if total_slots else 0.0,
            "model_x_camerapair_selections": int(all_models_combinations[:, :, m, :].sum()),
        }

    with open(os.path.join(base_path, "ensemble_model_selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    readme_path = os.path.join(base_path, "ensemble_model_selection_summary.txt")
    with open(readme_path, "w") as f:
        f.write("How often each ensemble member was chosen by the selector.\n")
        f.write("'fraction of frames selected' = share of frames where the model was "
                "part of the winning subset (any camera-pair).\n\n")
        f.write("Overall (averaged over the 4 joint-groups):\n")
        for m in range(num_models):
            frac = summary["overall"][model_names[m]]["fraction_of_frames_selected"]
            f.write(f"  [{m}] {model_names[m]}: {frac:.3f}\n")
        f.write("\nPer joint-group (fraction of frames selected):\n")
        header = "  {:<40}".format("model") + "".join(
            "{:>14}".format(g) for g in ENSEMBLE_GROUP_NAMES) + "\n"
        f.write(header)
        for m in range(num_models):
            row = "  {:<40}".format(f"[{m}] {model_names[m]}")
            for g in range(num_groups):
                row += "{:>14.3f}".format(frames_selected[g, m] / num_frames if num_frames else 0.0)
            f.write(row + "\n")

    try:
        vis_dir = os.path.join(base_path, "model_selection_visualizations")
        Visualizer.visualize_models_selection(all_models_combinations,
                                              output_dir=vis_dir,
                                              model_labels=model_names)
    except Exception as e:
        print(f"model-selection visualization failed: {e}", flush=True)

    if all_frames_scores is not None:
        try:
            with open(os.path.join(base_path, "all_frames_scores.json"), "w") as f:
                json.dump(_strip_frames_scores(all_frames_scores), f)
        except Exception as e:
            print(f"saving all_frames_scores failed: {e}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <config_path>")
        sys.exit(1)
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
