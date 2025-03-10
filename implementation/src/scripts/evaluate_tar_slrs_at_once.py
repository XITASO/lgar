import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.utils.evaluation_utils import create_run_label_exp1, create_run_label_exp2, create_run_label_exp3, extract_number
from implementation.src.utils.tar_utils import create_tar_output_tar_slrs, get_tar_metrics, remove_tar_output_files

CONFIG_PATH = "config.json"

def main(name_label, sort: bool, is_lm_only: bool = False):
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    combined_metrics = {}
    folder_path = config.llm_client_output_directory_path
    subdir = [f.path.replace("\\", "/") for f in os.scandir(folder_path) if f.is_dir()][0]
    labels = [os.path.basename(f) for f in os.scandir(subdir) if f.is_dir()]
    if sort:
        labels = sorted(labels, key=extract_number)
    for label in labels:
        print(f"Label: {label}")
        if label == "0s" or label == "Llama3.3-70B" or label == "Llama3.3-70B (Ti+RQ)":
            continue
        res_file_path, label_file_path = create_tar_output_tar_slrs(folder_path, config, label, is_lm_only)
        tar_metrics = get_tar_metrics(label_file_path, res_file_path)
        remove_tar_output_files(res_file_path, label_file_path)
        combined_metrics[name_label(label)] = tar_metrics
    desired_order = ["MAP", "TNR@95%", "R@1%", "R@5%", "R@10%", "R@20%", "R@50%", "WSS@95%", "WSS@100%"]
    combined_metrics = pd.DataFrame(combined_metrics).T
    combined_metrics = combined_metrics.reindex(columns=desired_order)
    combined_metrics = combined_metrics.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
    combined_metrics.to_csv(f"{config.llm_client_output_directory_path}/{config.llm_client_output_directory_path.split("/")[-2]}_tar2019_avg_metrics.csv")


if __name__ == "__main__":
    main(create_run_label_exp3, False, False)