import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.utils.tar_utils import create_tar_output_files_random, create_tar_output_files_random_all_tar_slrs, get_tar_metrics, remove_tar_output_files

CONFIG_PATH = "config.json"

def main(folder_path: str, num_runs: int, csv_output_path: str, all_tar: bool):
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    all_metrics = []
    for i in range(num_runs):
        print(f"Current run: {i}")
        if all_tar:
            res_file_path, label_file_path = create_tar_output_files_random_all_tar_slrs(folder_path=folder_path, seed=i, config=config)
        else:
            res_file_path, label_file_path = create_tar_output_files_random(folder_path=folder_path, seed=i, config=config)
        tar_metrics = get_tar_metrics(label_file_path, res_file_path)
        all_metrics.append(tar_metrics)
        remove_tar_output_files(res_file_path, label_file_path)
    
    mean_metrics = {
        key: round(sum(metric[key] for metric in all_metrics) / num_runs, 2)
        for key in all_metrics[0]
    }

    desired_order = ["MAP", "TNR@95%", "R@1%", "R@5%", "R@10%", "R@20%", "R@50%", "WSS@95%", "WSS@100%"]
    df = pd.DataFrame([mean_metrics], columns=desired_order)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_output_path, index=False)


if __name__ == "__main__":
    main(folder_path="./implementation/data/tar2019/", num_runs=5, csv_output_path="./implementation/data/paper/all_tar2019_random.csv", all_tar=True)