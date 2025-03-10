import argparse
import os
import sys
import time
import pandas as pd
sys.path.append(os.getcwd())
from implementation.src.data.guo_data_provider import GuoDataProvider
from implementation.src.data.tar_data_provider import TarDataProvider
import re
from implementation.src.utils.file_utils import load_json_file, save_ranked_df_with_reranker
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.data.results_handler import ResultHandler
from implementation.src.data.synergy_data_provider import SynergyDataProvider
from implementation.src.utils.data_utils import create_ranking_pointwise
from implementation.src.utils.evaluation_utils import create_run_label_exp1, create_run_label_exp2, create_run_label_exp3, extract_prefix
from implementation.src.utils.tar_utils import create_tar_output_files, get_tar_metrics, remove_tar_output_files

CONFIG_PATH = "config.json"

def main(index, name_label):
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    config.llm_client_config.path_to_reranker = "path/to/monoT5_3B"
    config.llm_client_config.system_message_type = "system_message_basic"
    experiment = config.llm_client_output_directory_path

    # if index == 0:
    #     experiment += "2s/"
    # elif index == 1:
    #     experiment += "CoT/"
    # elif index == 2:
    #     experiment += "CoT_sc/"
    # elif index == 3:
    #     experiment += "2s_CoT/"
    # elif index == 4:
    #     experiment += "2s_CoT_sc/"
    # if index == 0:
    #     experiment += "Llama3.1-8B/"
    # elif index == 1:
    #     experiment += "Qwen2.5-32B/"
    # elif index == 2:
    #     experiment += "Qwen2.5-72B/"
    # elif index == 3:
    #     experiment += "Mistral-123B/"
    if index == 0:
        experiment += "0-1/"
    elif index == 1:
        experiment += "0-2/"
    elif index == 2:
        experiment += "0-4/"
    elif index == 3:
        experiment += "0-9/"
    elif index == 4:
        experiment += "0-14/"
    elif index == 5:
        experiment += "0-19/"
    # elif index == 6:
    #     experiment += "0-24/"
    # elif index == 7:
    #     experiment += "0-29/"
    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    sub_folder_paths = [f.path.replace("\\", "/") for f in os.scandir(experiment) if f.is_dir()]
    for run in sub_folder_paths:
        print(f"run: {run}")
        slr_name = extract_prefix(run)

        slr_infos_df = load_json_file(config.file_path_slr_infos)[slr_name]
        
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
        start_time = time.time()
        result_df = create_ranking_pointwise(result_df, slr_infos_df, config)
        total_time = time.time() - start_time

        save_ranked_df_with_reranker(result_df["id"].tolist(), config.llm_client_config.path_to_reranker, run + "ranked_df.json", config.llm_client_config.system_message_type, total_time)
        result_df = pd.DataFrame(result_df["id"])


    res_file_path, label_file_path = create_tar_output_files(folder_path=experiment, config=config, is_lm_only=False)
    tar_metrics = get_tar_metrics(label_file_path, res_file_path)
    remove_tar_output_files(res_file_path, label_file_path)
    mean_metrics_df = pd.Series(tar_metrics)

    result_handler = ResultHandler(config=config, folder_path=experiment)
    result_handler.store_mean_metrics_and_std(metrics=mean_metrics_df, std=None, tag=name_label(experiment.split("/")[-2]), with_std=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', type=int, required=True, help='Index of the job')
    args = parser.parse_args()
    main(args.index, create_run_label_exp1)