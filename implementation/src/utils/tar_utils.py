import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from implementation.src.data.results_handler import ResultHandler
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.data.synergy_data_provider import SynergyDataProvider
from implementation.src.data.guo_data_provider import GuoDataProvider
from implementation.src.data.tar_data_provider import TarDataProvider
from implementation.src.utils.evaluation_utils import extract_prefix
from implementation.src.utils.file_utils import load_json_file, scan_folder_for_csv
from implementation.src.config.config import Config
from implementation.src.scripts_tar.tar_eval_2018 import main

CONFIG_PATH = "config.json"

def get_tar_metrics(label_file_path: str, result_file_path: str):
    metric_dict = main(2, results_file=result_file_path, qrel_file=label_file_path)

    # filter out metrics that we want
    metrics = {
        "MAP": metric_dict["ap"] * 100,
        "WSS@95%": metric_dict["wss_95"] * 100,
        "WSS@100%": metric_dict["wss_100"] * 100,
        "R@1%": metric_dict["R@1%"] * 100,
        "R@5%": metric_dict["R@5%"] * 100,
        "R@10%": metric_dict["R@10%"] * 100,
        "R@20%": metric_dict["R@20%"] * 100,
        "R@50%": metric_dict["R@50%"] * 100,
        "TNR@95%": metric_dict["tnr95"] * 100,
    }
    return metrics

def write_to_output_file(folder_path: str, data_results, data_label):
    res_file_path = f'{folder_path}output.res'
    with open(res_file_path, 'w') as file:
        for line in data_results:
            file.write(line + '\n')
    label_file_path = f'{folder_path}label_file'
    with open(label_file_path, "w") as file:
        for line in data_label:
            file.write(line + '\n')
    return res_file_path, label_file_path

def create_tar_output_files(folder_path: str, config: Config, is_lm_only: bool):
    """
    Create the output.res and label_file files for the TREC Ad-hoc Retrieval track evaluation.

    :param folder_path: The path to the folder containing the runs (folder which contains runs of which the results should be averaged).
    :param config: The configuration object.
    :param is_lm_only: Whether the evaluation is of runs with only language models.
    :return: The paths to the output.res and label_file files.
    """
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    sub_folder_paths = [f.path.replace("\\", "/") for f in os.scandir(folder_path) if f.is_dir()]
    data_results, data_label = [], []
    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    for run in sub_folder_paths:
        print(f"run: {run}")
        slr_name = extract_prefix(run)
        slr_df = data_provider.create_dataframe(f"{slr_name}.csv")
        # load the dataset (potential relevant papers for the SLR)
        result_df = None
        if is_lm_only:
            json_file_path = run + "/logfile.json"
            result_handler = ResultHandler(config=config, folder_path=run)
            result_df = result_handler.process_ranked_df_json_to_df(json_file_path)
        else:
            json_file_path = run + "/" + "ranked_df.json"
            json_data = load_json_file(json_file_path)
            result_df = pd.DataFrame(json_data["ids"], columns=["id"])
        result_df["id"] = result_df["id"].apply(int)
        slr_df_sub = pd.merge(result_df["id"], slr_df, on="id", how="left", validate="one_to_one")
        for i in range(len(slr_df_sub)):
            data_results.append(f"{slr_name} 0 {slr_df_sub.iloc[i]["id"]} {i + 1} {float(-(i + 1))} pubmed")
            data_label.append(f"{slr_name} 0 {slr_df_sub.iloc[i]["id"]} {slr_df_sub.iloc[i]["label"]}")
    return write_to_output_file(folder_path, data_results, data_label)


def create_tar_output_files_random_all_tar_slrs(folder_path: str, seed: int, config: Config):
    """
    Create the output.res and label_file files for the TREC Ad-hoc Retrieval track evaluation of a random baseline. Each file of the folder will occur once in the created output files.

    :param folder_path: Path to the folder containing the csv files of the slr or a dataset collection.
    :param seed: seed for shuffling the documents in each slr dataframe.
    :param config: The configuration object.
    """
    subfolder_paths = [f.path.replace("\\", "/") for f in os.scandir(folder_path) if f.is_dir()]
    data_results, data_label = [], []
    for subfolder in subfolder_paths:
        subfolder = subfolder + "/"
        files = scan_folder_for_csv(folder_path=subfolder)
        for slr_paths in files:
            file_name = os.path.basename(slr_paths)
            data_provider = TarDataProvider(config)
            slr_df = data_provider.create_dataframe(file_name)
            slr_df = slr_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            for i in range(len(slr_df)):
                data_results.append(f"{file_name.split(".csv")[0]} 0 {slr_df.iloc[i]["id"]} {i + 1} {float(-(i + 1))} pubmed")
                data_label.append(f"{file_name.split(".csv")[0]} 0 {slr_df.iloc[i]["id"]} {slr_df.iloc[i]["label"]}")
    return write_to_output_file(folder_path, data_results, data_label)

def create_tar_output_files_random(folder_path: str, seed: int, config: Config):
    """
    Create the output.res and label_file files for the TREC Ad-hoc Retrieval track evaluation of a random baseline. Each file of the folder will occur once in the created output files.

    :param folder_path: Path to the folder containing the csv files of the slr or a dataset collection.
    :param seed: seed for shuffling the documents in each slr dataframe.
    :param config: The configuration object.
    """

    files = scan_folder_for_csv(folder_path=folder_path)
    data_results, data_label = [], []
    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    for slr_paths in files:
        file_name = os.path.basename(slr_paths)
        slr_df = data_provider.create_dataframe(file_name)
        slr_df = slr_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        for i in range(len(slr_df)):
            data_results.append(f"{file_name.split(".csv")[0]} 0 {slr_df.iloc[i]["id"]} {i + 1} {float(-(i + 1))} pubmed")
            data_label.append(f"{file_name.split(".csv")[0]} 0 {slr_df.iloc[i]["id"]} {slr_df.iloc[i]["label"]}")
    return write_to_output_file(folder_path, data_results, data_label)

def create_tar_output_tar_slrs(folder_path: str, config: Config, label: str, is_lm_only: bool):
    """
    Create the output.res and label_file files for the TREC Ad-hoc Retrieval track evaluation.

    :param folder_path: The path to the folder containing the runs (folder which contains runs of which the results should be averaged).
    :param config: The configuration object.
    :param label: The label of the subofolder.
    :param is_lm_only: Whether the evaluation is of runs with only language models.
    :return: The paths to the output.res and label_file files.
    """
    def get_label_folder(sub_folder_paths, label):
        for path in sub_folder_paths:
            if label in path:
                return path
        return None
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    dataset_dirs = [f.path.replace("\\", "/") for f in os.scandir(folder_path) if f.is_dir()]
    data_results, data_label = [], []
    for experiment in dataset_dirs:
        sub_folder_paths = [f.path.replace("\\", "/") for f in os.scandir(experiment) if f.is_dir()]
        label_folder = get_label_folder(sub_folder_paths, label)
        sub_folder_paths = [f.path.replace("\\", "/") for f in os.scandir(label_folder) if f.is_dir()]
        if experiment.split("/")[-1] == "synergy":
            continue
        for run in sub_folder_paths:
            print(f"run: {run}")
            slr_name = extract_prefix(run)
            config.folder_path_slrs = "./implementation/data/tar2019/" + experiment.split("/")[-1] + "/"

            # initialization
            data_provider = SynergyDataProvider(config)
            if "tar2019" in config.folder_path_slrs:
                data_provider = TarDataProvider(config)
            elif "guo" in config.folder_path_slrs:
                data_provider = GuoDataProvider(config)
            slr_df = data_provider.create_dataframe(f"{slr_name}.csv")
            result_df = None

            if is_lm_only:
                json_file_path = run + "/logfile.json"
                result_handler = ResultHandler(config=config, folder_path=run)
                result_df = result_handler.process_ranked_df_json_to_df(json_file_path)
            else:
                json_file_path = run + "/" + "ranked_df.json"
                json_data = load_json_file(json_file_path)
                result_df = pd.DataFrame(json_data["ids"], columns=["id"])
            result_df["id"] = result_df["id"].apply(int)
            slr_df_sub = pd.merge(result_df["id"], slr_df, on="id", how="left", validate="one_to_one")
            for i in range(len(slr_df_sub)):
                data_results.append(f"{slr_name} 0 {slr_df_sub.iloc[i]["id"]} {i + 1} {float(-(i + 1))} pubmed")
                data_label.append(f"{slr_name} 0 {slr_df_sub.iloc[i]["id"]} {slr_df_sub.iloc[i]["label"]}")
    return write_to_output_file(folder_path, data_results, data_label)

def remove_tar_output_files(res_file_path: str, label_file_path: str):
    """
    Remove the output.res and label_file files for the TREC Ad-hoc Retrieval track evaluation.

    :param res_file_path: The path to the output.res file.
    :param label_file_path: The path to the label_file file.
    """
    os.remove(res_file_path)
    os.remove(label_file_path)