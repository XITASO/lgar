import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from implementation.src.data.guo_data_provider import GuoDataProvider
from implementation.src.utils.evaluation_utils import extract_prefix
from implementation.src.data.tar_data_provider import TarDataProvider
import re
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.data.results_handler import ResultHandler
from implementation.src.data.synergy_data_provider import SynergyDataProvider
from implementation.src.utils.tar_utils import get_tar_metrics, remove_tar_output_files, write_to_output_file

CONFIG_PATH = "config.json"

def main(folder_path: str, num_runs: int, csv_output_path: str, label: str, all_tar: bool):
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    all_metrics = []
    folder_paths = [folder_path]
    if all_tar:
        folder_paths = [f.path + "/" for f in os.scandir(folder_path) if f.is_dir()]
    document_count = 0
    for folder_path_ in folder_paths:
        folder_name = folder_path_.split("/")[-2]
        if all_tar and folder_name == "synergy":
            continue
        if all_tar and "tar2019" not in config.folder_path_slrs:
            config.folder_path_slrs = config.folder_path_slrs.replace(config.folder_path_slrs.split("/")[-2], f"tar2019/{folder_name}")
        else:
            config.folder_path_slrs = config.folder_path_slrs.replace(config.folder_path_slrs.split("/")[-2], folder_name)
        config.file_path_slr_infos = config.file_path_slr_infos.replace(config.file_path_slr_infos.split("/")[-2], folder_name)
        if all_tar:
            folder_path_new = [f.path for f in os.scandir(folder_path_) if f.is_dir() and label in f.name][0]
        else:
            folder_path_new = folder_path_
        sub_folder_paths = [f.path.replace("\\", "/") for f in os.scandir(folder_path_new) if f.is_dir()]
        data_provider = SynergyDataProvider(config)
        if "tar2019" in config.folder_path_slrs:
            data_provider = TarDataProvider(config)
        elif "guo" in config.folder_path_slrs:
            data_provider = GuoDataProvider(config)
        for run in sub_folder_paths:
            document_count += 1
            print(f"run: {run}")
            slr_name = extract_prefix(run)
            config.folder_path_slrs = f"{'/'.join(config.folder_path_slrs.split('/')[:-1])}/"
            
            # load the dataset (potential relevant papers for the SLR)
            slr_df = data_provider.create_dataframe(f"{slr_name}.csv")
            run = run + "/"
            # initiate result handler
            result_handler = ResultHandler(
                config=config,
                folder_path=run,
            )
            file_names = [f for f in os.listdir(run) if re.match(r'log_file_\d+\.json', f)]
            result_df = None
            if len(file_names) == 1:
                file_path = run + file_names[0]
                result_df, _, _ = result_handler.process_json_to_dataframe(file_path)
            else:
                result_df, _ = result_handler.create_self_consistency_df(file_names)
            if "abstract" not in result_df.columns:
                abstracts = slr_df[["abstract", "id", "title"]]
                result_df["id"] = result_df["id"].astype(int)
                result_df = pd.merge(abstracts, result_df, on="id", validate="one_to_one")
            result_df["relevance_of_paper"] = result_df["relevance_of_paper"].astype(float)

            result_df.loc[result_df["relevance_of_paper"] == -1, "relevance_of_paper"] = result_df["relevance_of_paper"].mean()
            result_df = result_df.sort_values(by="relevance_of_paper", ascending=False, kind="mergesort") # use stable sort
            result_df_grouped = result_df.groupby("relevance_of_paper")
            unique_vals = result_df["relevance_of_paper"].unique()
            all_metrics_of_one_slr = []
            for i in range(num_runs):
            # create random ranking
                new_df = pd.DataFrame()
                data_results, data_label = [], []
                for val in unique_vals:
                    new_group = result_df_grouped.get_group(val)
                    if new_group.shape[0] > 1:
                        new_df = pd.concat([new_df, new_group.sample(frac=1, random_state=i).reset_index(drop=True)])
                    else:
                        new_df = pd.concat([new_df, new_group])
                for i in range(len(new_df)):
                    data_results.append(f"{slr_name} 0 {new_df.iloc[i]["id"]} {i + 1} {float(-(i + 1))} pubmed")
                    data_label.append(f"{slr_name} 0 {new_df.iloc[i]["id"]} {new_df.iloc[i]["ground_truth"]}")
                res_file_path, label_file_path = write_to_output_file(folder_path=folder_path_new, data_results=data_results, data_label=data_label)
                tar_metrics = get_tar_metrics(label_file_path, res_file_path)
                all_metrics_of_one_slr.append(tar_metrics)
                remove_tar_output_files(res_file_path, label_file_path)
            mean_metrics = {
                key: sum(metric[key] for metric in all_metrics_of_one_slr) / num_runs
                for key in all_metrics_of_one_slr[0]
            }
            all_metrics.append(mean_metrics)

    overall_mean_metrics = {
        key: round(sum(metric[key] for metric in all_metrics) / document_count, 2)
        for key in all_metrics[0]
    }
    desired_order = ["MAP", "TNR@95%", "R@1%", "R@5%", "R@10%", "R@20%", "R@50%", "WSS@95%", "WSS@100%"]
    df = pd.DataFrame([overall_mean_metrics], columns=desired_order)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    main(folder_path="./implementation/data/paper/scales/qualitative/0-19", num_runs=5, csv_output_path="./implementation/data/paper/scales/qualitative/0-19/random_reranking.csv", label="0-19", all_tar=False)