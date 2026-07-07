import os
import sys
abspath = os.path.abspath(__file__)
code_directory = os.path.dirname(os.path.dirname(abspath))
sys.path.append(code_directory)
import torch
from torchsummary import summary
from datetime import date
import time
import Preprocessor
import Datasets
import Network
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from utils import TrainConfig, optimizer_from_string, create_train_run_folders, save_training_code, show_interest_points_with_index
import Callbacks
from constants import CONFIGURATION_FILE_NAME


N = 0
C = 1
H = 2
W = 3
REPORT_EVERY = 100

def arrange_loaded_checkpoint(general_configuration: TrainConfig):
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
                 general_configuration: TrainConfig,
                 base_run_directory,
                 device):
        self.device = device
        if general_configuration.debug_mode:
            self.batches_per_epoch = 1

        self.general_configuration = general_configuration
        self.base_run_directory = base_run_directory
        self.ensure_resume_compatibility()
        self.val_fraction = self.general_configuration.get_val_fraction()
        self.preprocessor = Preprocessor.Preprocessor(self.general_configuration)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.num_epochs = self.general_configuration.get_num_epochs()
        self.checkpoint_load_path = self.general_configuration.get_resume_training_checkpoint_path()
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
        self.network = Network.Network(self.general_configuration, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels)
        self.model = self.network.get_model()
        self.model.to(self.device)
        self.model_input_shape = (1, self.number_of_input_channels, self.img_size[1], self.img_size[2])

        summary(self.model, input_size=(self.number_of_input_channels, self.img_size[1], self.img_size[2]))

        self.loss_function = self.general_configuration.configure_loss()
        self.optimizer = optimizer_from_string[self.general_configuration.optimizer_as_string](
            self.model.parameters(),
            lr=self.general_configuration.learning_rate,
            eps=self.general_configuration.optimizer_epsilon
            )
        
        # Cosine annealing: smoothly decay the LR from its initial value down
        # to eta_min over the whole run, following a half-cosine curve. Unlike
        # ReduceLROnPlateau this is schedule-based rather than tied to the
        # background-dominated validation MSE, so it cannot starve the LR early
        # when that metric briefly plateaus (which previously froze training
        # ~1/3 of the way through). T_max is the planned number of epochs.
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.general_configuration.reduce_lr_min_lr
        )

        if self.checkpoint_load_path:
            self._load_checkpoint(self.checkpoint_load_path)

        self.train_box, self.train_confmap, self.val_box, self.val_confmap, _, _ = self.train_val_split_resume()
        viz_sample_list = (self.val_box[:self.general_configuration.how_many_visualizations], self.val_confmap[:self.general_configuration.how_many_visualizations])

        # show_interest_points_with_index(viz_sample_list[0], viz_sample_list[1], save_directory='.', filename="viz_sample_points.png")

        print("img_size:", self.img_size, flush=True)
        print("num_output_channels:", self.num_output_channels, flush=True)

        self.callbacks = Callbacks.ModelCallbacks(
                                        model=self.model,
                                        base_directory=base_run_directory,
                                        viz_sample_list=viz_sample_list,
                                        validation=(self.val_box, self.val_confmap),
                                        training=(self.train_box, self.train_confmap)
                                        )
    
    def ensure_resume_compatibility(self):
        original_config_path = os.path.join(self.base_run_directory, CONFIGURATION_FILE_NAME)
        original_configuration = TrainConfig(config_path=original_config_path)
        if original_configuration.get_val_fraction() != self.general_configuration.get_val_fraction():
            raise ValueError("Validation fraction in the resumed configuration does not match the original configuration.")
        if original_configuration.batch_size != self.general_configuration.batch_size:
            raise ValueError("Batch size in the resumed configuration does not match the original configuration.")
        if original_configuration.model_type != self.general_configuration.model_type:
            raise ValueError("Model type in the resumed configuration does not match the original configuration.")
        if original_configuration.kernel_size != self.general_configuration.kernel_size:
            raise ValueError("Kernel size in the resumed configuration does not match the original configuration.")
        if original_configuration.num_base_filters != self.general_configuration.num_base_filters:
            raise ValueError("Number of base filters in the resumed configuration does not match the original configuration.")
        if original_configuration.num_blocks != self.general_configuration.num_blocks:
            raise ValueError("Number of encoder-decoder blocks in the resumed configuration does not match the original configuration.")
        if original_configuration.loss_function_as_string != self.general_configuration.loss_function_as_string:
            raise ValueError("Loss function in the resumed configuration does not match the original configuration.")
        return

    def create_visualization_dataset(self):
        pass

    def save_model_as_scripted(self):
        file_name = "best_model.pt"
        file_path = os.path.join(self.base_run_directory, file_name)
        self.model.eval()
        try:
            device = next(self.model.parameters()).device
            dummy_input = torch.randn(self.model_input_shape).to(device)
            scripted_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(scripted_model, file_path)
            print(f'Model successfully saved as scripted model to {file_path}', flush=True)
        except Exception as e:
            print(f'Error saving model as scripted: {e}', flush=True)
        finally:
            self.model.train()


    def _save_checkpoint(self, epoch, best=False):
        if not best:
            ckp = {
                "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
            }

            save_directory = os.path.join(self.base_run_directory, 'weights')
            file_name = f"model_epoch_{epoch + 1}.pth"
            save_path = os.path.join(save_directory, file_name)
            torch.save(ckp, save_path)
            print(f'Epoch {epoch+1} - Training checkpoint was save to {save_path}', flush=True)
        else:
            self.save_model_as_scripted()
            txt_file_path = os.path.join(self.base_run_directory, "best_model_info.txt")
            with open(txt_file_path, 'w') as f:
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Best Validation Loss: {self.best_val_loss:.6f}\n")

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"]
        new_state_dict = {}
        for key, value in state_dict.items():
            # Replace 'model' with 'layers' in the keys to match current architecture
            new_key = key.replace("encoder.model", "encoder.layers")
            new_key = new_key.replace("decoder.model", "decoder.layers")
            new_state_dict[new_key] = value
        state_dict = new_state_dict

        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

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
        # --- CRITICAL DEBUG BLOCK ---
        if torch.isnan(loss) or torch.isinf(loss):
            print("CRITICAL ERROR: Loss turned to NaN/Inf during training step!")
            print(f"Loss value: {loss.item()}")
            
            # Check if logits caused it
            if torch.isnan(outputs).any():
                print(" -> Cause: Model outputs were already NaN before loss.")
            else:
                print(" -> Cause: The Loss function math failed (likely log(0)).")
            
            # STOP execution so you can see the error
            raise ValueError("Training stopped due to NaN loss.")
        # -----------------------------

        loss.backward()
        
        # --- OPTIONAL: Gradient Clipping (Highly Recommended for JSD) ---
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # --------------------------------------------------------------
        self.optimizer.step()
        return loss.item()

    def do_one_epoch(self, epoch_number, train_loader, val_loader):
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
        logs['train loss'] = avg_train_loss
        
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
            self._save_checkpoint(epoch=epoch_number, best=True)

        logs['validation loss'] = average_val_loss

        # CosineAnnealingLR steps per-epoch and takes no metric argument.
        self.lr_scheduler.step()
        logs['lr'] = self.lr_scheduler.get_last_lr()[0]

        self.callbacks.on_epoch_end(epoch=epoch_number, logs=logs)

    def train(self):
        augmentor = Datasets.Augmentor(self.general_configuration)
        train_set = Datasets.Dataset(self.train_box, self.train_confmap, augmentor.get_transforms())
        val_set = Datasets.Dataset(self.val_box, self.val_confmap)
        train_loader = Datasets.prepare_dataloader(train_set, self.general_configuration.batch_size)
        val_loader = Datasets.prepare_dataloader(val_set, self.general_configuration.batch_size)

        training_start_time = time.time()
        self.callbacks.on_train_start()

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            self.do_one_epoch(epoch_number=epoch, train_loader=train_loader, val_loader=val_loader)
            if self.general_configuration.save_every > 0 and \
                    epoch % self.general_configuration.save_every == 0:
                self._save_checkpoint(epoch=epoch)

        training_end_time = time.time()

        elapsed_time = training_end_time - training_start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Training completed in {int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2} (hh:mm:ss)', flush=True)
    
    def train_val_split_resume(self, shuffle=True):
        """ 
        Splits datasets into train and validation sets. 
        Handles saving/loading of splits for reproducible resume training.
        """
        # 1. Check if we are resuming from a previous run
        resume_dir = self.general_configuration.resume_training_directory
        
        # Define the standard filename for the split
        split_filename = "train_val_split.npz"

        # --- PATH A: RESUME TRAINING ---
        if resume_dir is not None:
            split_file_path = os.path.join(resume_dir, split_filename)
            
            if os.path.exists(split_file_path):
                print(f"[Trainer] Resuming: Loading data split from {split_file_path}")
                # Load the indices
                data = np.load(split_file_path)
                train_idx = data['train_idx']
                val_idx = data['val_idx']
            else:
                raise FileNotFoundError(
                    f"[Trainer] Resume directory provided, but {split_filename} not found in {resume_dir}. "
                    "Cannot guarantee reproducible data split."
                )

        # --- PATH B: FRESH TRAINING ---
        else:
            # Standard calculation of size
            val_size = int(np.round(len(self.box) * self.val_fraction))
            all_idx = np.arange(len(self.box))
            
            # Shuffle if requested
            if shuffle:
                np.random.shuffle(all_idx)
            
            # Slice the indices
            val_idx = all_idx[:val_size]
            train_idx = all_idx[val_size:]
            
            # SAVE THE SPLIT
            save_path = os.path.join(self.base_run_directory, split_filename)
            np.savez(save_path, train_idx=train_idx, val_idx=val_idx)
            print(f"[Trainer] Created new split and saved to {save_path}")

        # Apply the indices to the data
        # Note: Using .copy() is often safer to ensure memory continuity, 
        # but strictly optional depending on your RAM constraints.
        return (self.box[train_idx], self.confmaps[train_idx], 
                self.box[val_idx], self.confmaps[val_idx], 
                train_idx, val_idx)

def training_main(
                general_configuration,
                base_run_directory,
                use_gpu
               ):
    if use_gpu:
        device = torch.device(f'cuda:{0}')
        print(f"Running on {torch.cuda.get_device_name(device=device)}", flush=True)
    else:
        device = torch.device("cpu")
        print(f"Running on CPU", flush=True)
    try:
        trainer = Trainer(
            general_configuration=general_configuration,
            base_run_directory=base_run_directory,
            device=device
        )
        trainer.train()        
    except Exception as e:
        import traceback
        print(f"Exception during training: {e}", flush=True)
        traceback.print_exc()

def main():
    overall_start_time = time.time()
    config_path = sys.argv[1] if len(sys.argv) > 1 else exit("Please provide a config file.")
    print(f"Using config file: {config_path}", flush=True)
    general_configuration = TrainConfig(config_path=config_path)

    base_output_directory = arrange_loaded_checkpoint(general_configuration=general_configuration)

    if not base_output_directory:
        date_str = date.today().strftime('%b %d')
        run_tag = general_configuration.get_run_tag()
        run_name = f"{general_configuration.model_type}_{run_tag}_{date_str}" if run_tag \
            else f"{general_configuration.model_type}_{date_str}"
        base_output_directory = create_train_run_folders(
            base_output_directory=general_configuration.get_base_output_directory(),
            run_name=run_name,
            original_config_file=general_configuration.get_config_file())
    save_training_code(base_output_directory)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        use_gpu = True
    else:
        use_gpu = False
        print("⚠️ GPU not available, using CPU instead.", flush=True)


    training_main(
        general_configuration=general_configuration,
        base_run_directory=base_output_directory,
        use_gpu=use_gpu
    )

    # Total wall-clock for the whole run: preprocessing + setup + training.
    total_elapsed = time.time() - overall_start_time
    hours, rem = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Total run time (preprocessing + training): '
          f'{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2} (hh:mm:ss)', flush=True)

if __name__ == "__main__":
    main()