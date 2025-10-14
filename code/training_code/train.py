import os
import sys
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
import random
import torch
from datetime import date
import time
import Preprocessor
import Datasets
import Network
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from utils import *
import Callbacks
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce, ReduceOp, get_world_size

N = 0
C = 1
H = 2
W = 3
REPORT_EVERY = 100

def ddp_setup(rank, world_size, port, use_gpu):
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    os.environ["MASTER_PORT"] = port
    backend = 'nccl' if use_gpu else "gloo"
    init_process_group(backend=backend, rank=rank, world_size=world_size)

def arrange_loaded_checkpoint(general_configuration: Config):
    '''
    Check if resuming from checkpoint, if so, check that both directory and file exist.
    If not resuming, create new run directory.
    returns the base output directory to use if resuming. If not resuming, returns None.
    '''
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
                 gpu_id,
                 device):
        self.device = device
        if general_configuration.debug_mode:
            self.batches_per_epoch = 1

        self.base_run_directory = base_run_directory
        self.val_fraction = general_configuration.get_val_fraction()
        self.gpu_id = gpu_id
        self.preprocessor = Preprocessor.Preprocessor(general_configuration)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.num_epochs = general_configuration.get_num_epochs()
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
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            self.model = DDP(self.model)

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
        if self.gpu_id == 0:
            print("img_size:", self.img_size, flush=True)
            print("num_output_channels:", self.num_output_channels, flush=True)
            show_sample_channels(self.viz_sample[0], self.base_run_directory, filename="viz_sample.png")
            show_interest_points_with_index(self.viz_sample[0], self.viz_sample[1], self.base_run_directory, filename="viz_sample_points.png")
        # test_transforms(self.train_box[0], self.train_confmap[0], self.base_run_directory, [
        #     Augmentor.Scale()
        # ])

        self.callbacks = Callbacks.ModelCallbacks(
                                        model=self.model,
                                        base_directory=base_run_directory,
                                        viz_sample=self.viz_sample,
                                        validation=(self.val_box, self.val_confmap)
                                        )
        self.general_configuration = general_configuration

    def _save_checkpoint(self, epoch, best=False):

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
        print(f'Epoch {epoch+1} - Training checkpoint was save to {save_path}', flush=True)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']+1})", flush=True)
        
    def train_step(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def do_one_epoch(self, epoch_number, train_loader, val_loader):
        if self.gpu_id == 0:
            self.callbacks.on_epoch_begin(epoch=epoch_number)

        whole_epoch_train_loss_sum = 0.0
        step_count = 0

        logs = {}

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            loss = self.train_step(inputs, labels)

            whole_epoch_train_loss_sum += loss
            step_count += 1

        avg_train_loss = whole_epoch_train_loss_sum / step_count
        avg_train_loss_tensor = torch.tensor([avg_train_loss], device=self.device)
        reduce(avg_train_loss_tensor, dst=0, op=ReduceOp.SUM)
        if self.gpu_id == 0:
            general_avg_train_loss = avg_train_loss_tensor / get_world_size()
            logs['train loss'] = general_avg_train_loss.item()
            
        running_val_loss = 0.0
        step_count = 0
        self.model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                running_val_loss += loss.item()
                step_count += 1

        average_val_loss = running_val_loss / step_count

        if average_val_loss < self.best_val_loss:
            self.best_val_loss = average_val_loss
            if self.gpu_id == 0:
                self._save_checkpoint(epoch=epoch_number, best=True)

        average_val_loss_tensor = torch.tensor([average_val_loss], device=self.device)
        reduce(average_val_loss_tensor, dst=0, op=ReduceOp.SUM)
        if self.gpu_id == 0:
            general_avg_validation_loss = average_val_loss_tensor / get_world_size()
            # print(f'[ALL GPUs] | Epoch {epoch_number}/{self.num_epochs} | Train Loss: {general_avg_train_loss.item():.4f}, Validation Loss: {general_avg_validation_loss.item():.8f}')
            logs['validation loss'] = general_avg_validation_loss.item()

        self.lr_scheduler.step(average_val_loss)
        logs['lr'] = self.lr_scheduler.get_last_lr()[0]

        if self.gpu_id == 0:
            self.callbacks.on_epoch_end(epoch=epoch_number, logs=logs)

    def train(self):
        augmentor = Datasets.Augmentor(self.general_configuration)
        train_set = Datasets.Dataset(self.train_box, self.train_confmap, augmentor.get_transforms())
        val_set = Datasets.Dataset(self.val_box, self.val_confmap)
        train_loader = Datasets.prepare_dataloader(train_set, self.general_configuration.batch_size)
        val_loader = Datasets.prepare_dataloader(val_set, self.general_configuration.batch_size)

        training_start_time = time.time()
        if self.gpu_id == 0:
            self.callbacks.on_train_start()

        for epoch in range(self.start_epoch, self.num_epochs):
            train_loader.sampler.set_epoch(epoch)
            self.do_one_epoch(epoch_number=epoch, train_loader=train_loader, val_loader=val_loader)
            if self.gpu_id == 0 and \
                self.general_configuration.save_every > 0 and \
                    epoch % self.general_configuration.save_every == 0:
                self._save_checkpoint(epoch=epoch)

        training_end_time = time.time()

        elapsed_time = training_end_time - training_start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.gpu_id == 0:
            print(f'Training completed in {int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2} (hh:mm:ss)', flush=True)

        
    def train_val_split(self, shuffle=True):
        """ Splits datasets into train and validation sets. """
        val_size = int(np.round(len(self.box) * self.val_fraction))
        idx = np.arange(len(self.box))
        if shuffle:
            np.random.shuffle(idx)
        val_idx = idx[:val_size]
        idx = idx[val_size:]
        return self.box[idx], self.confmaps[idx], self.box[val_idx], self.confmaps[val_idx], idx, val_idx

def joint_main(rank,
               world_size,
               general_configuration,
               base_run_directory,
               port,
               use_gpu):
    ddp_setup(rank=rank, world_size=world_size, port=port, use_gpu=use_gpu)
    if use_gpu:
        device = torch.device(f'cuda:{rank}')
        print(f"[Rank {rank}] Running on {torch.cuda.get_device_name(device=device)}", flush=True)
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Running on CPU", flush=True)
    try:
        trainer = Trainer(
            general_configuration=general_configuration,
            base_run_directory=base_run_directory,
            gpu_id=rank,
            device=device
        )
        trainer.train()        
    except Exception as e:
        import traceback
        print(f"[Rank {rank}] Exception during training: {e}", flush=True)
        traceback.print_exc()
    finally:
        destroy_process_group()

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else exit("Please provide a config file.")
    print(f"Using config file: {config_path}", flush=True)
    general_configuration = Config(config_path=config_path)

    base_output_directory = arrange_loaded_checkpoint(general_configuration=general_configuration)

    if not base_output_directory:
        run_name = f"{general_configuration.model_type}_{date.today().strftime('%b %d')}"
        base_output_directory = create_run_folders(
            base_output_directory=general_configuration.get_base_output_directory(),
            run_name=run_name,
            original_config_file=general_configuration.get_config_file())
        save_training_code(base_output_directory)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        world_size = torch.cuda.device_count()
        use_gpu = True
    else:
        world_size = 1
        use_gpu = False
        print("⚠️ GPU not available, using CPU instead.", flush=True)

    port = str(random.randint(10000, 20000))  # pick free port
    mp.spawn(joint_main, args=(world_size,
                               general_configuration,
                               base_output_directory,
                               port,
                               use_gpu), nprocs=world_size)
    


if __name__ == "__main__":
    main()