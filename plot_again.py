import pandas as pd
from utils import plot_graph, get_proper_df
import os
import train_config

if __name__ == '__main__':
    config = train_config.config
    plot_path = os.path.join(config["model_name"], "Plot and CSV")
    loss_df = pd.read_csv(os.path.join(config["model_name"], "metrics.csv"), index_col= 0)
    print("Columns of the logged info are:")
    print(list(loss_df["metric"].unique()))
    if config['paired']:
        metric_list = config['metrics']
        # Check if the metrics asked are supported or not
        for metric in metric_list:
            assert metric.lower() in ['ssim', 'fid', 'psnr'], f"{metric} is not supported"
        # Lower it just incase
        metric_list = [metric.lower() for metric in metric_list]
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
    os.makedirs(plot_path, exist_ok= True) # Create the folder if needed
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