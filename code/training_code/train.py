
import json
import os
import shutil
import sys
from datetime import date
import preprocessor
from Datasets import *
import Network
import numpy as np
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from utils import *
from Callbacks import ModelCallbacks

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log what device is being used
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available, using CPU instead.")

N = 0
C = 1
H = 2
W = 3

class Trainer:
    def __init__(self, config_path):
        with open(config_path) as CF:
            config = json.load(CF)
            self.config = config
            self.batch_size = config['batch size']
            self.num_epochs = config['epochs']
            self.batches_per_epoch = config['batches per epoch']
            self.val_fraction = config['val fraction']
            self.loss_function = loss_from_string[config["loss function"]]()
            self.learning_rate = config["learning rate"]
            self.optimizer_as_string = config["optimizer"]
            self.reduce_lr_factor = config["reduce lr factor"]
            self.reduce_lr_patience = config["reduce lr patience"]
            self.reduce_lr_min_delta = config["reduce lr min delta"]
            self.reduce_lr_cooldown = config["reduce lr cooldown"]
            self.reduce_lr_min_lr = config["reduce lr min lr"]
            self.base_output_directory = config["base output directory"]
            self.viz_idx = 1
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
        self.img_size = (self.box.shape[C], self.box.shape[H], self.box.shape[W])
        self.number_of_input_channels = self.box.shape[C]
        self.num_output_channels = self.confmaps.shape[C]
        self.network = Network.Network(config, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels)
        self.model = self.network.get_model()

        self.optimizer = optimizer_from_string[self.optimizer_as_string](
            self.model.parameters(),
            lr=self.learning_rate
            )
        
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            cooldown=2,
            threshold=0.01
        )

        self.train_box, self.train_confmap, self.val_box, self.val_confmap, _, _ = self.train_val_split()
        self.validation = (self.val_box, self.val_confmap)
        self.viz_sample = (self.val_box[self.viz_idx], self.val_confmap[self.viz_idx])
        print("img_size:", self.img_size)
        print("num_output_channels:", self.num_output_channels)

        # test_transforms(self.train_box[0], self.train_confmap[0], self.run_path, [
        #     Augmentor.Scale(scale_range=(0.4, 1.6))
        # ])

        self.callbacks = ModelCallbacks(config, self.model, self.run_path, self.viz_sample, (self.val_box, self.val_confmap))

    def do_one_epoch(self, epoch_number, train_loader, val_loader):
        self.callbacks.on_epoch_begin(epoch=epoch_number)

        running_trainig_loss = 0.0
        last_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_trainig_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_trainig_loss / 100
                print(f"[{i + 1}, {self.batches_per_epoch}] training loss: {last_loss:.9f}")
                running_trainig_loss = 0.0

        average_train_loss = running_trainig_loss / len(train_loader)

        running_val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                running_val_loss += loss.item()
        
        average_val_loss = running_val_loss / (i + 1)

        self.lr_scheduler.step(average_val_loss)

        logs = {
            'train loss': average_train_loss,
            'validation loss': average_val_loss,
            'lr': self.lr_scheduler.get_last_lr()[0]
        }
        self.callbacks.on_epoch_end(epoch=epoch_number, logs=logs)

    def train(self):
        augmentor = Augmentor(self.config)
        train_set = Dataset(self.train_box, self.train_confmap, augmentor.get_transforms())
        val_set = Dataset(self.val_box, self.val_confmap)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        
        self.callbacks.on_train_start()
        for epoch in range(self.num_epochs):
            self.model.train()
            self.do_one_epoch(epoch_number=epoch, train_loader=train_loader, val_loader=val_loader)

        
    def train_val_split(self, shuffle=True):
        """ Splits datasets into train and validation sets. """
        val_size = int(np.round(len(self.box) * self.val_fraction))
        idx = np.arange(len(self.box))
        if shuffle:
            np.random.shuffle(idx)
        val_idx = idx[:val_size]
        idx = idx[val_size:]
        return self.box[idx], self.confmaps[idx], self.box[val_idx], self.confmaps[val_idx], idx, val_idx

    def create_run_folders(self):
        """ Creates folders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_directory, self.run_name)
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
    trainer.train()


if __name__ == "__main__":
    main()