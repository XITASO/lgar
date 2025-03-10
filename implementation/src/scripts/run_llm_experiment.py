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

    print("Creating an LLM client ...")
    if index == 0:
        print("Llama3.1-8B-Instruct")
        config.llm_client_config.path_to_model = "path/to/Meta-Llama-3.1-8B-Instruct"
        config.llm_client_config.name_of_model = "Llama3.1-8B"
    elif index == 1:
        print("Qwen2.5-32B-Instruct")
        config.llm_client_config.path_to_model = "path/to/Qwen2.5-32B-Instruct"
        config.llm_client_config.name_of_model = "Qwen2.5-32B"
    elif index == 2:
        print("Qwen2.5-72B-Instruct")
        config.llm_client_config.path_to_model = "path/to/Qwen2.5-72B-Instruct"
        config.llm_client_config.name_of_model = "Qwen2.5-72B"
    elif index == 3:
        print("Mistral-Large-Instruct-2411")
        config.llm_client_config.path_to_model = "path/to/Mistral-Large-Instruct-2411"
        config.llm_client_config.name_of_model = "Mistral-123B"
    config.llm_client_output_directory_path = config.llm_client_output_directory_path + config.llm_client_config.name_of_model + "/"
    client = LocalClient(config=config)

    print("Sending test prompt to verify correct setup: ", end="\n")
    client.send_test_prompt()

    print("LLM client created successfully.", end="\n")
    client = client.client
    print("Prompting Technique Experiment")
    config.relevance_lower_value = "0"
    config.relevance_upper_value = "19"
    config.llm_client_config.system_message_type = "system_message_rq"
    config.llm_client_config.path_to_reranker = ""
    config.llm_client_config.temperature = 0
    config.llm_client_config.prompting_technique = "zero_shot"
    config.llm_client_config.number_consistency_path = 0
    config.llm_client_config.is_few_shot = False
    config.llm_client_config.ordering_few_shot_examples = ""
    for slr in slr_files:
        print(f"current file: {slr}")
        slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
        run_experiment_with_evaluation(config, client, 0, slr_name)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', type=int, required=True, help='Index of the job')
    args = parser.parse_args()
    main(args.index)
