import os
import sys
from datetime import date
from preprocessor import Preprocessor
from Datasets import *
from utils import *
import Network
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from utils import *
from Callbacks import ModelCallbacks
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 0
C = 1
H = 2
W = 3

def ddp_setup(rank, world_size, use_gpu):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = 'nccl' if use_gpu else "gloo"
    init_process_group(backend=backend, rank=rank, world_size=world_size)

def arrange_loaded_checkpoint(general_configuration: Config):
    file = general_configuration.get_resume_training_checkpoint_path()
    directory = general_configuration.get_resume_training_directory()
    if (file and not directory) or (directory and not file):
        exit("resume training directory, checkpoint file only one of them is missing")
    if file and directory:
        both_exist = os.path.exists(file) and os.path.exists(directory)
        if not both_exist:
            exit("resume training directory or checkpoint file doesn't exist")
    return directory

class Trainer:
    def __init__(self,
                 general_configuration: Config,
                 base_run_directory,
                 gpu_id):

        if general_configuration.debug_mode:
            self.batches_per_epoch = 1

        self.base_run_directory = base_run_directory
        self.val_fraction = general_configuration.get_val_fraction()
        self.gpu_id = gpu_id
        self.preprocessor = Preprocessor(general_configuration)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.checkpoint_load_path = general_configuration.get_resume_training_checkpoint_path()
        if self.checkpoint_load_path and len(self.checkpoint_load_path) > 0 and not os.path.exists(self.checkpoint_load_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at: {self.checkpoint_load_path}"
            )


        # Do preprocessing according to the model type
        self.preprocessor.do_preprocess()
        self.box, self.confmaps = self.preprocessor.get_box(), self.preprocessor.get_confmaps()

        # Get the right CNN architecture
        self.img_size = (self.box.shape[C], self.box.shape[H], self.box.shape[W])
        self.number_of_input_channels = self.box.shape[C]
        self.num_output_channels = self.confmaps.shape[C]
        self.network = Network.Network(general_configuration, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels)
        self.model = self.network.get_model()
        self.model.to(device)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.loss_function = loss_from_string[general_configuration.loss_function_as_string]()
        self.optimizer = optimizer_from_string[general_configuration.optimizer_as_string](
            self.model.parameters(),
            lr=general_configuration.learning_rate
            )
        
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=general_configuration.reduce_lr_factor,
            patience=general_configuration.reduce_lr_patience,
            cooldown=general_configuration.reduce_lr_cooldown,
            threshold=0.01,
            min_lr=general_configuration.reduce_lr_min_lr
        )

        if self.checkpoint_load_path:
            self._load_checkpoint(self.checkpoint_load_path)

        self.train_box, self.train_confmap, self.val_box, self.val_confmap, _, _ = self.train_val_split()
        self.validation = (self.val_box, self.val_confmap)
        self.viz_sample = (self.val_box[general_configuration.viz_idx], self.val_confmap[general_configuration.viz_idx])
        print("img_size:", self.img_size)
        print("num_output_channels:", self.num_output_channels)

        # test_transforms(self.train_box[0], self.train_confmap[0], self.run_path, [
        #     Augmentor.Scale(scale_range=(0.4, 1.6))
        # ])

        self.callbacks = ModelCallbacks(
                                        model=self.model,
                                        base_directory=base_run_directory,
                                        viz_sample=self.viz_sample,
                                        validation=(self.val_box, self.val_confmap)
                                        )
        self.general_configuration = general_configuration

    def _save_checkpoint(self, epoch, best=False):
        # ckp = self.model.module.state_dict()

        ckp = {
            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
        }

        save_directory = os.path.join(self.base_run_directory, 'weights')
        file_name = f"best_model.pt" if best else f"model_epoch_{epoch + 1}.pt"
        save_path = os.path.join(save_directory, file_name)
        torch.save(ckp, save_path)
        print(f'Epoch {epoch+1} - Training checkpoint was save to {save_path}')

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']+1})")
        

    def do_one_epoch(self, epoch_number, train_loader, val_loader):
        self.callbacks.on_epoch_begin(epoch=epoch_number)

        running_trainig_loss = 0.0
        last_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # outputs = outputs.to(device)
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
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                running_val_loss += loss.item()
        
        average_val_loss = running_val_loss / (i + 1)

        if average_val_loss < self.best_val_loss:
            self.best_val_loss = average_val_loss
            if self.gpu_id == 0:
                self._save_checkpoint(epoch=epoch_number, best=True)

        self.lr_scheduler.step(average_val_loss)

        logs = {
            'train loss': average_train_loss,
            'validation loss': average_val_loss,
            'lr': self.lr_scheduler.get_last_lr()[0]
        }
        self.callbacks.on_epoch_end(epoch=epoch_number, logs=logs)

    def train(self):
        augmentor = Augmentor(self.general_configuration)
        train_set = Dataset(self.train_box, self.train_confmap, augmentor.get_transforms())
        val_set = Dataset(self.val_box, self.val_confmap)
        train_loader = prepare_dataloader(train_set, self.general_configuration.batch_size)
        val_loader = prepare_dataloader(val_set, self.general_configuration.batch_size)
        
        self.callbacks.on_train_start()
        for epoch in range(self.start_epoch, self.general_configuration.num_epochs):
            self.model.train()
            self.do_one_epoch(epoch_number=epoch, train_loader=train_loader, val_loader=val_loader)
            if self.gpu_id == 0 and \
                self.general_configuration.save_every > 0 and \
                    epoch % self.general_configuration.save_every == 0:
                self._save_checkpoint(epoch=epoch)

        
    def train_val_split(self, shuffle=True):
        """ Splits datasets into train and validation sets. """
        val_size = int(np.round(len(self.box) * self.val_fraction))
        idx = np.arange(len(self.box))
        if shuffle:
            np.random.shuffle(idx)
        val_idx = idx[:val_size]
        idx = idx[val_size:]
        return self.box[idx], self.confmaps[idx], self.box[val_idx], self.confmaps[val_idx], idx, val_idx

def joint_main(rank, world_size, general_configuration, base_run_directory, use_gpu):
    ddp_setup(rank, world_size, use_gpu)
    try:
        trainer = Trainer(
            general_configuration=general_configuration,
            base_run_directory=base_run_directory,
            gpu_id=rank
        )
        trainer.train()        
    except Exception as e:
        import traceback
        print(f"[Rank {rank}] Exception during training: {e}")
        traceback.print_exc()
    finally:
        destroy_process_group()

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else exit("Please provide a config file.")
    print(f"Using config file: {config_path}")
    general_configuration = Config(config_path=config_path)

    base_output_directory = arrange_loaded_checkpoint(general_configuration=general_configuration)

    if not base_output_directory:
        run_name = f"{general_configuration.model_type}_{date.today().strftime('%b %d')}"
        base_output_directory = create_run_folders(
            base_output_directory=general_configuration.get_base_output_directory(),
            run_name=run_name,
            original_config_file=general_configuration.get_config_file())
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        world_size = torch.cuda.device_count()
        use_gpu = True
    else:
        world_size = 1
        use_gpu = False
        print("⚠️ GPU not available, using CPU instead.")

    
    mp.spawn(joint_main, args=(world_size, general_configuration, base_output_directory, use_gpu), nprocs=world_size)
    


if __name__ == "__main__":
    main()