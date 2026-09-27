import argparse
import json
import os
import shutil
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
from make_heldout_split import md5_of
from constants import CONFIGURATION_FILE_NAME, LATEST_CHECKPOINT_FILE_NAME, TRAIN_VAL_INDICES_FILE_NAME


N = 0
C = 1
H = 2
W = 3
REPORT_EVERY = 100

class Trainer:
    def __init__(self,
                 general_configuration: TrainConfig,
                 base_run_directory,
                 device,
                 resume_checkpoint=None):
        self.device = device
        if general_configuration.debug_mode:
            self.batches_per_epoch = 1

        self.general_configuration = general_configuration
        self.base_run_directory = base_run_directory
        self.preprocessor = Preprocessor.Preprocessor(self.general_configuration)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.num_epochs = self.general_configuration.get_num_epochs()
        self.checkpoint_load_path = resume_checkpoint

        # Do preprocessing according to the model type
        self.preprocessor.do_preprocess()
        self.box, self.confmaps = self.preprocessor.get_box(), self.preprocessor.get_confmaps()

        # Get the right CNN architecture
        self.img_size = (self.box.shape[C], self.box.shape[H], self.box.shape[W])
        self.number_of_input_channels = self.box.shape[C]
        self.num_output_channels = self.confmaps.shape[C]
        self.network = Network.Network(self.general_configuration, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels,
                                       num_cams=self.preprocessor.cams_per_sample)
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

        self.train_box, self.train_confmap, self.val_box, self.val_confmap = self.train_val_split()
        viz_sample_list = (self.val_box[:self.general_configuration.how_many_visualizations], self.val_confmap[:self.general_configuration.how_many_visualizations])

        # show_interest_points_with_index(viz_sample_list[0], viz_sample_list[1], save_directory='.', filename="viz_sample_points.png")

        print("img_size:", self.img_size, flush=True)
        print("num_output_channels:", self.num_output_channels, flush=True)

        self.callbacks = Callbacks.ModelCallbacks(
                                        model=self.model,
                                        base_directory=base_run_directory,
                                        viz_sample_list=viz_sample_list,
                                        validation=(self.val_box, self.val_confmap),
                                        training=(self.train_box, self.train_confmap),
                                        num_cams=self.preprocessor.cams_per_sample
                                        )
    
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

            # Keep a single rolling resume checkpoint instead of one file per
            # epoch: the latest model+optimizer+scheduler+epoch is all the
            # resume feature needs, and this stops weights/ from growing
            # unbounded (which previously filled the disk and killed runs).
            # Write to a temp file then atomically replace, so a kill mid-write
            # can't corrupt the only resume point.
            save_directory = os.path.join(self.base_run_directory, 'weights')
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, LATEST_CHECKPOINT_FILE_NAME)
            tmp_path = save_path + ".tmp"
            torch.save(ckp, tmp_path)
            os.replace(tmp_path, save_path)
            print(f'Epoch {epoch+1} - Training checkpoint was saved to {save_path}', flush=True)
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

        if self.start_epoch >= self.num_epochs:
            print(f"All {self.num_epochs} planned epochs are already done; nothing to resume.", flush=True)
            return

        training_start_time = time.time()
        self.callbacks.on_train_start(start_epoch=self.start_epoch)

        # "epochs" in the config is the run's TOTAL, and the cosine schedule
        # spans exactly that many, so a resumed run continues to the same end.
        for epoch in range(self.start_epoch, self.num_epochs):
            self.do_one_epoch(epoch_number=epoch, train_loader=train_loader, val_loader=val_loader)
            if self.general_configuration.save_every > 0 and \
                    epoch % self.general_configuration.save_every == 0:
                self._save_checkpoint(epoch=epoch)

        training_end_time = time.time()

        elapsed_time = training_end_time - training_start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Training completed in {int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2} (hh:mm:ss)', flush=True)
    
    def split_indices(self):
        """Train/val sample indices from the frozen split (make_heldout_split.py).

        Preprocessing turns one labelled frame into several samples (its two
        wings, and for 3-camera models its camera subsets), so the split is
        made per FRAME: every sample goes wherever its source frame is.
        Test frames go nowhere -- no model trains on them or picks its
        checkpoint with them, so every model can be scored on them.

        Every error here is fatal: a model that drew its own split, or saw the
        test frames, could no longer be compared with the others."""
        groups = self.preprocessor.sample_group_ids
        n_samples = len(self.box)
        split_path = self.general_configuration.get_split_file()
        split = np.load(split_path)
        frame_split = split["frame_split"]
        meta = json.loads(str(split["meta"]))
        if groups is None or len(groups) != n_samples:
            raise ValueError(f"split file {split_path} needs a source frame per "
                             f"sample, and this model type does not report one")
        if len(frame_split) != self.preprocessor.num_frames:
            raise ValueError(f"split file covers {len(frame_split)} frames, "
                             f"the dataset has {self.preprocessor.num_frames}")
        data_path = self.general_configuration.get_data_path()
        if md5_of(data_path) != meta["dataset_md5"]:
            raise ValueError(f"split file {split_path} was made for a different "
                             f"dataset than {data_path}")
        labels = meta["labels"]
        sample_side = frame_split[groups]
        train_idx = np.flatnonzero(sample_side == labels["train"])
        val_idx = np.flatnonzero(sample_side == labels["val"])
        np.random.shuffle(train_idx)
        n_test = int((sample_side == labels["test"]).sum())
        print(f"[Trainer] split from {split_path}: "
              f"{(frame_split == labels['train']).sum()} train / "
              f"{(frame_split == labels['val']).sum()} val / "
              f"{(frame_split == labels['test']).sum()} test frames -> "
              f"{len(train_idx)} / {len(val_idx)} samples, {n_test} test samples "
              f"held out", flush=True)
        return train_idx, val_idx

    def train_val_split(self):
        """The train and validation samples. A fresh run takes them from the
        split file and saves the exact indices (and a copy of the file) in the
        run folder; a resumed run reloads those indices, so it continues on
        precisely the samples -- in the order -- it started with."""
        indices_path = os.path.join(self.base_run_directory, TRAIN_VAL_INDICES_FILE_NAME)
        if self.checkpoint_load_path:
            data = np.load(indices_path)
            train_idx, val_idx = data['train_idx'], data['val_idx']
            print(f"[Trainer] resuming on the saved split {indices_path}: "
                  f"{len(train_idx)} / {len(val_idx)} samples", flush=True)
        else:
            train_idx, val_idx = self.split_indices()
            np.savez(indices_path, train_idx=train_idx, val_idx=val_idx)
            shutil.copy(self.general_configuration.get_split_file(), self.base_run_directory)
        return (self.box[train_idx], self.confmaps[train_idx],
                self.box[val_idx], self.confmaps[val_idx])

def training_main(
                general_configuration,
                base_run_directory,
                use_gpu,
                resume_checkpoint=None
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
            device=device,
            resume_checkpoint=resume_checkpoint
        )
        trainer.train()        
    except Exception as e:
        import traceback
        print(f"Exception during training: {e}", flush=True)
        traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a pose-estimation model.",
        epilog="Fresh run:  train.py <config.json>\n"
               "Resume:     train.py --resume <run folder>   (uses the run's own saved "
               "configuration.json and continues to its planned last epoch)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("config", nargs="?", help="training configuration json")
    source.add_argument("--resume", metavar="RUN_FOLDER",
                        help="continue the run in this folder from its last checkpoint")
    return parser.parse_args()

def main():
    overall_start_time = time.time()
    args = parse_args()

    resume_checkpoint = None
    if args.resume:
        # A resume needs nothing but the run folder: it holds the exact config
        # the run started with, the checkpoint, and the split indices. Taking
        # the config from there means it cannot disagree with the run.
        base_output_directory = os.path.abspath(args.resume)
        config_path = os.path.join(base_output_directory, CONFIGURATION_FILE_NAME)
        resume_checkpoint = os.path.join(base_output_directory, "weights", LATEST_CHECKPOINT_FILE_NAME)
        for needed in (config_path, resume_checkpoint,
                       os.path.join(base_output_directory, TRAIN_VAL_INDICES_FILE_NAME)):
            if not os.path.exists(needed):
                exit(f"cannot resume {base_output_directory}: {needed} is missing")
        print(f"Resuming {base_output_directory}", flush=True)
        general_configuration = TrainConfig(config_path=config_path)
        # Keep the code the run started with; add this code beside it.
        save_training_code(base_output_directory,
                           folder_name=f"training code (resumed {date.today().isoformat()})")
    else:
        print(f"Using config file: {args.config}", flush=True)
        general_configuration = TrainConfig(config_path=args.config)
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
        use_gpu=use_gpu,
        resume_checkpoint=resume_checkpoint
    )

    # Total wall-clock for the whole run: preprocessing + setup + training.
    total_elapsed = time.time() - overall_start_time
    hours, rem = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Total run time (preprocessing + training): '
          f'{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2} (hh:mm:ss)', flush=True)

if __name__ == "__main__":
    main()