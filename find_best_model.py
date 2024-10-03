# Imports
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

# Pipeline Imports
import infer_config
from UnpairedDataset import UnpairedDataset
from PairedDataset import PairedDataset

# from JoinedPairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import save_paired_image, save_unpaired_image, load_weights, convert_to_255
from metrics import get_metric_class, calc_metric

######
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

if __name__ == "__main__":
    # Get the config file
    config = infer_config.config
    root = config["data_path"]
    mode = config["sub_fold"]
    if config["paired"]:
        metric_list = config["metrics"]
        # Check if the metrics asked are supported or not
        for metric in metric_list:
            assert metric.lower() in [
                "ssim",
                "fid",
                "psnr",
            ], f"{metric} is not supported"
        # Lower it just incase
        metric_list = [metric.lower() for metric in metric_list]
    else:
        raise NotImplementedError("Metrics for Unpaired Data are not implemented.")
    metric_list = ["psnr"]  # temp hardcode
    # Set paired or unpaired
    if config["paired"]:
        test_ds = PairedDataset(root, mode)
    else:
        test_ds = UnpairedDataset(root, mode)
    # Create Dataloader
    test_dl = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    model_list = [f for f in os.listdir(config["model_name"]) if f.endswith(".ckpt")]
    print(f"\nTotal {len(model_list)} models found\n")
    total_met_list = []
    for model_name in model_list:
        # Model Loaded
        print(
            f"Loading model from -> {os.path.join(config['model_name'],model_name)}\n"
        )
        model = CycleGan.load_from_checkpoint(
            os.path.join(config["model_name"], model_name)
        )
        # Set model to 'cuda' and evaluation mode
        model.eval()
        model.to("cuda")
        # Create Classes for Metric computation
        metric_dict = {}
        for metric in metric_list:
            metric_dict[f"{metric}_A"] = get_metric_class(metric).to("cuda")
            metric_dict[f"{metric}_B"] = get_metric_class(metric).to("cuda")
        # Iterate through batches
        for i, batch in enumerate(tqdm(test_dl, total=len(test_dl))):
            imgA, imgB = batch["A"].to("cuda"), batch["B"].to("cuda")
            with torch.no_grad():
                fakeB = model.genX(imgA)
                fakeA = model.genY(imgB)
            # Convert to [0, 255] for calculation
            imgA_255 = convert_to_255(imgA)
            imgB_255 = convert_to_255(imgB)
            fakeB_255 = convert_to_255(fakeB)
            fakeA_255 = convert_to_255(fakeA)
            # Compute Metrics
            for metric in metric_list:
                metric_dict[f"{metric}_A"] = calc_metric(
                    metric_dict[f"{metric}_A"], metric, fakeA_255, imgA_255
                )
                metric_dict[f"{metric}_B"] = calc_metric(
                    metric_dict[f"{metric}_B"], metric, fakeB_255, imgB_255
                )
            # break
        # Print the metrics
        print(f"{'#'*20}")
        if metric_list:
            print(f"\nMetrics Calculated:\n")
        for metric in metric_list:
            metric_A, metric_B = metric_dict[f"{metric}_A"], metric_dict[f"{metric}_B"]
            print(f"{metric.upper()}_A = {metric_A.compute():.3f}")
            print(f"{metric.upper()}_B = {metric_B.compute():.3f}")
            print(
                f"{metric.upper()}_full = {(metric_A.compute() + metric_B.compute())/2:.3f}\n"
            )
        metric_to_append = (metric_A.compute() + metric_B.compute()) / 2
        total_met_list.append(metric_to_append)
    df = pd.DataFrame(
        {"Model": model_list, metric_list[0]: total_met_list},
        columns=["Model", total_met_list],
    )
    df.to_csv(os.path.join(config["model_name"], "best_model.csv"))
    ###################
    print("Inference Completed!")
