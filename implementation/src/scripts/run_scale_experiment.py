import argparse
import sys
import os
sys.path.append(os.getcwd())
from implementation.src.client.local_client import LocalClient
from implementation.src.utils.file_utils import scan_folder_for_csv
import torch
from implementation.src.utils.experiment_utils import run_experiment_with_evaluation
from implementation.src.config.config_loader import ConfigLoader
CONFIG_PATH = "config.json"


def main(index):
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config

    folder_path = os.path.dirname(config.folder_path_slrs) + "/"
    slr_files = scan_folder_for_csv(folder_path)
    output_path = config.llm_client_output_directory_path

    print("Creating an LLM client ...")
    client = LocalClient(config=config)

    print("Sending test prompt to verify correct setup: ", end="\n")
    client.send_test_prompt()

    print("LLM client created successfully.", end="\n")
    client = client.client

    for slr in slr_files:
        print(f"current file: {slr}")

        config.relevance_lower_value = "0"
        config.relevance_upper_value = "1"
        config.llm_client_config.temperature = 0
        config.llm_client_config.prompting_technique = "zero_shot"
        config.llm_client_config.system_message_type = "system_message_rq"
        config.llm_client_config.number_consistency_path = 0
        config.llm_client_config.is_few_shot = False
        config.llm_client_config.path_to_reranker = ""
        config.llm_client_output_directory_path = output_path + "0-1/"
        if index == 0:
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()

            print("Run 2: zero-shot, sm with rqs, scale: 0-2, re-ranker: default")
            config.relevance_upper_value = "2"
            config.llm_client_output_directory_path = output_path + "0-2/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()
        if index == 1:
            print("Run 3: zero-shot, sm with rqs, scale: 0-4, re-ranker: default")
            config.relevance_upper_value = "4"
            config.llm_client_output_directory_path = output_path + "0-4/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()

            print("Run 4: zero-shot, sm with rqs, scale: 0-9, re-ranker: default")
            config.relevance_upper_value = "9"
            config.llm_client_output_directory_path = output_path + "0-9/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()
        if index == 2:
            print("Run 5: zero-shot, sm with rqs, scale: 0-14, re-ranker: default")
            config.relevance_upper_value = "14"
            config.llm_client_output_directory_path = output_path + "0-14/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()

            print("Run 6: zero-shot, sm with rqs, scale: 0-19, re-ranker: default")
            config.relevance_upper_value = "19"
            config.llm_client_output_directory_path = output_path + "0-19/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()

        if index == 3:
            print("Run 7: zero-shot, sm with rqs, scale: 0-24, re-ranker: default")
            config.relevance_upper_value = "24"
            config.llm_client_output_directory_path = output_path + "0-24/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()

            print("Run 8: zero-shot, sm with rqs, scale: 0-29, re-ranker: default")
            config.relevance_upper_value = "29"
            config.llm_client_output_directory_path = output_path + "0-29/"
            slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
            run_experiment_with_evaluation(config, client, 0, slr_name)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', type=int, required=True, help='Index of the job')
    args = parser.parse_args()
    main(args.index)
