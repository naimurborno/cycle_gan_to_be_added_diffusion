# train.py
# Imports
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import os
import shutil

# Pipeline Imports
import train_config
from UnpairedDataset import UnpairedDataset
from PairedDataset import PairedDataset
#from JoinedPairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import plot_graph, tflog2pandas, get_proper_df
from tqdm import tqdm

# Filter the warning.
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

if __name__ == "__main__":
    # Get the config file
    config = train_config.config
    config["n_epochs"] = config['n_lin_epoch'] + config['n_dec_epoch']
    # Set Path for data
    root = config["data_path"]
    if config['paired']:
        metric_list = config['metrics']
        # Check if the metrics asked are supported or not
        for metric in metric_list:
            assert metric.lower() in ['ssim', 'fid', 'psnr'], f"{metric} is not supported"
        # Lower it just incase
        metric_list = [metric.lower() for metric in metric_list]
    else:
        raise NotImplementedError("Metrics for Unpaired Data are not implemented.")
    # Set paired or unpaired
    if config["paired"]:
        train_ds = PairedDataset(root, "Train")
    else:
        train_ds = UnpairedDataset(root, "Train")
    # Create Loader
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)

    # Resuming Check; disabled `global_step_offset` now
    if config["resume_ckpt"] is not None:
        resume_ckpt = os.path.join(config["model_name"], config["resume_ckpt"])
        assert os.path.exists(resume_ckpt), f"{resume_ckpt} file does not exist"
    else:
        resume_ckpt = config["resume_ckpt"]  # it will be None so no resume

    # Pipeline Callbacks
    logger = pl.loggers.TensorBoardLogger(
        save_dir=config["model_name"], name="", version="", default_hp_metric=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["model_name"],
        filename="cyclegan-{epoch:05d}-{step}",
        every_n_epochs=config["save_freq"],
        save_last=True,
        verbose=True,
        save_top_k=-1,
    )
    # Instantiate Model

    # model = CycleGan(config)
    if resume_ckpt is None:
        model = CycleGan(config)
    else:
        model = CycleGan.load_from_checkpoint(os.path.join(config['model_name'],config['resume_ckpt']))

    # Instantiate Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["n_epochs"],
        log_every_n_steps=1,
        default_root_dir=config["model_name"],
        logger=logger,
        callbacks=[checkpoint_callback],
        # profiler="pytorch" # Enable if needed
    )

    # Train Model
    tqdm(trainer.fit(model, train_dl, ckpt_path=resume_ckpt))
    # print (trainer.profiler.summary()) # Enable if needed

    # Copy config file
    shutil.copy(
        src="train_config.py", dst=os.path.join(config["model_name"], "train_config.py")
    )
    # Loss plot
    loss_path = config["model_name"]  # folderpath where tensorboard data is stored
    plot_path = os.path.join(config["model_name"], "Plot and CSV")
    os.makedirs(plot_path, exist_ok= True)
    loss_df = tflog2pandas(loss_path)
    loss_df.to_csv(
        os.path.join(config["model_name"], "metrics.csv")
    )  # os.path.join(config['model_name'],'metrics.csv')
    print(
        f"Loss and Metric Plots are saved in -> {plot_path}"
    )
    # Plots
    metrics = {
        "gen_loss_epoch": "Generator Loss",
        "dis_loss_epoch": "Discriminator Loss",
        "id_loss_epoch": "Identity Loss",
        "cyc_loss_epoch": "Cycle Consistency Loss",
    }  # loss type : Loss title
    for metric in metric_list:
        metrics[f'{metric}_A_epoch'] = f"{metric.upper()} between Predicted and Actual A"
        metrics[f'{metric}_B_epoch'] = f"{metric.upper()} between Predicted and Actual B"

    for metric in metrics.keys():
        metric_df = get_proper_df(loss_df, metric=metric, step="epoch")
        save_path = os.path.join(plot_path, f"{metric}.png")
        plot_graph(
            metric_df["epoch"],
            metric_df["value"],
            xlabel="# Epoch",
            ylabel="Loss",
            title=metrics[metric],
            save_path=save_path,
        )
