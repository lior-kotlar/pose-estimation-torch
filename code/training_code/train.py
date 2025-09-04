
import json
import os
import shutil
import sys
from datetime import date
import preprocessor
import Network

class Trainer:
    def __init__(self, config_path):
        with open(config_path) as C:
            config = json.load(C)
            self.config = config
            self.batch_size = config['batch size']
            self.num_epochs = config['epochs']
            self.batches_per_epoch = config['batches per epoch']
            self.val_fraction = config['val fraction']
            self.base_output_path = config["base output directory"]
            self.model_type = config["model type"]
            self.debug_mode = bool(config["debug mode"])
            self.preprocessor = preprocessor.Preprocessor(config)

        if self.debug_mode:
            self.batches_per_epoch = 1

        self.run_name = f"{self.model_type}_{date.today().strftime('%b %d')}"
        self.clean = False
        self.run_path = self.create_run_folders()
        self.save_configuration()

        # Do preprocessing according to the model type
        self.preprocessor.do_preprocess()
        self.box, self.confmaps = self.preprocessor.get_box(), self.preprocessor.get_confmaps()

        # Get the right CNN architecture
        self.img_size = self.box.shape[1:]
        self.number_of_input_channels = self.box.shape[-1]
        self.num_output_channels = self.confmaps.shape[-1]
        self.network = Network.Network(config, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels)
        self.model = self.network.get_model()

    def create_run_folders(self):
        """ Creates folders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_path, self.run_name)
        if not self.clean:
            initial_run_path = run_path
            i = 1
            while os.path.exists(run_path):
                run_path = "%s_%02d" % (initial_run_path, i)
                i += 1
        if os.path.exists(run_path):
            shutil.rmtree(run_path)
        os.makedirs(run_path)
        os.makedirs(os.path.join(run_path, "weights"))
        os.makedirs(os.path.join(run_path, "viz_pred"))
        os.makedirs(os.path.join(run_path, "histograms"))
        os.makedirs(os.path.join(run_path, "l2_histograms_per_point"))
        print("Created folder:", run_path)
        code_dir_path = os.path.join(run_path, "training code")
        os.makedirs(code_dir_path)
        for file_name in os.listdir('.'):
            if file_name.endswith('.py'):
                full_file_name = os.path.join('.', file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, code_dir_path)
                    print(f"Copied {full_file_name} to {code_dir_path}")
        return run_path
    
    def save_configuration(self):
        with open(f"{self.run_path}/configuration.json", 'w') as file:
            json.dump(self.config, file, indent=4)
    
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else exit("Please provide a config file.")
    print(f"Using config file: {config_path}")
    trainer = Trainer(config_path)


if __name__ == "__main__":
    main()