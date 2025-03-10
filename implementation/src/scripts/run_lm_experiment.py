import argparse
import sys
import os
sys.path.append(os.getcwd())
from implementation.src.data.guo_data_provider import GuoDataProvider
from implementation.src.client.bm25_client import BM25Client
from implementation.src.client.monoBERT_client import MonoBERTClient
from implementation.src.client.monoT5_client import MonoT5Client
from implementation.src.client.colbert_client import ColBERTClient
from implementation.src.data.synergy_data_provider import SynergyDataProvider
from implementation.src.utils.file_utils import scan_folder_for_csv
import torch
from implementation.src.utils.experiment_utils import run_experiment_bert_0_shot
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.data.tar_data_provider import TarDataProvider


CONFIG_PATH = "./config.json"

def main(index):
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config

    config.llm_client_config.system_message_type = "system_message_basic"
    add_to_path = " (T)/"
    if index == 1:
        add_to_path = " (T+R)/"
        config.llm_client_config.system_message_type = "system_message_rq"

    folder_path = os.path.dirname(config.folder_path_slrs) + "/"
    slr_files = scan_folder_for_csv(folder_path)
    output_path = config.llm_client_output_directory_path
    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    for slr in slr_files:
        print(f"current file: {slr}")
        slr_name = os.path.basename(slr).split(".csv")[0]
        # load the dataset (potential relevant papers for the SLR)
        slr_df = data_provider.create_dataframe(f"{slr_name}.csv")


        print("Run 0: BM25")
        client = BM25Client(model_path="", ids=slr_df["id"].astype(str).tolist())
        config.llm_client_output_directory_path = output_path + "BM25" + add_to_path
        run_experiment_bert_0_shot(config, client, "BM25", 0, slr_name)
        torch.cuda.empty_cache()

        print("Run 1: 0-shot ColBERT")
        client = ColBERTClient(model_path="path/to/colbert")
        config.llm_client_output_directory_path = output_path + "ColBERT" + add_to_path
        run_experiment_bert_0_shot(config, client, "BM25", 0, slr_name)
        torch.cuda.empty_cache()

        print("Run 2: 0-shot monoBERT")
        client = MonoBERTClient(model_path="path/to/monoBERT")
        config.llm_client_output_directory_path = output_path + "monoBERT" + add_to_path
        run_experiment_bert_0_shot(config, client, "BM25", 0, slr_name)
        torch.cuda.empty_cache()

        print("Run 3: 0-shot monoT5")
        client = MonoT5Client(model_path="path/to/monoT5_3B")
        config.llm_client_output_directory_path = output_path + "monoT5_3B" + add_to_path
        run_experiment_bert_0_shot(config, client, "BM25", 0, slr_name)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', type=int, required=True, help='Index of the job')
    args = parser.parse_args()
    main(args.index)
