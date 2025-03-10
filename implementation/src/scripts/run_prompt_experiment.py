import argparse
import sys
import os
sys.path.append(os.getcwd())
from implementation.src.client.local_client import LocalClient
from implementation.src.utils.file_utils import scan_folder_for_csv
import torch
from implementation.src.utils.experiment_utils import create_few_shot_examples, run_experiment_with_evaluation
from implementation.src.config.config_loader import ConfigLoader
CONFIG_PATH = "config.json"


def main(index, generate_examples: bool):
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
    print("Prompting Technique Experiment")
    for slr in slr_files:
        print(f"current file: {slr}")
        slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
        config.relevance_lower_value = "0"
        config.relevance_upper_value = "19"
        config.llm_client_config.system_message_type = "system_message_rq"
        config.llm_client_config.path_to_reranker = ""
        if generate_examples:
            if index == 0:
                print("zero-shot")
                config.llm_client_config.temperature = 0
                config.llm_client_config.prompting_technique = "zero_shot"
                config.llm_client_config.number_consistency_path = 0
                config.llm_client_config.is_few_shot = False
                config.llm_client_config.ordering_few_shot_examples = ""
                create_few_shot_examples(client, 0, slr_name)
                torch.cuda.empty_cache()
            elif index == 1:
                print("CoT")
                config.llm_client_config.temperature = 0
                config.llm_client_config.prompting_technique = "CoT"
                config.llm_client_config.number_consistency_path = 0
                config.llm_client_config.is_few_shot = False
                config.llm_client_config.ordering_few_shot_examples = ""
                create_few_shot_examples(client, 0, slr_name)
                torch.cuda.empty_cache()
        else:
            if index == 0:
                print("2-shot")
                config.llm_client_config.temperature = 0
                config.llm_client_config.prompting_technique = "zero_shot"
                config.llm_client_config.number_consistency_path = 0
                config.llm_client_config.is_few_shot = True
                config.llm_client_config.ordering_few_shot_examples = "PN"
                config.llm_client_output_directory_path = output_path + "2s/"
                slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
                run_experiment_with_evaluation(config, client, 0, slr_name)
                torch.cuda.empty_cache()
            elif index == 1:
                print("CoT")
                config.llm_client_config.temperature = 0
                config.llm_client_config.prompting_technique = "CoT"
                config.llm_client_config.number_consistency_path = 0
                config.llm_client_config.is_few_shot = False
                config.llm_client_config.ordering_few_shot_examples = ""
                config.llm_client_output_directory_path = output_path + "CoT/"
                slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
                run_experiment_with_evaluation(config, client, 0, slr_name)
                torch.cuda.empty_cache()
            elif index == 2:
                print("CoT (n=3)")
                config.llm_client_config.temperature = 0.5
                config.llm_client_config.prompting_technique = "CoT"
                config.llm_client_config.number_consistency_path = 2
                config.llm_client_config.is_few_shot = False
                config.llm_client_config.ordering_few_shot_examples = ""
                config.llm_client_output_directory_path = output_path + "CoT_sc/"
                slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
                run_experiment_with_evaluation(config, client, 0, slr_name)
                torch.cuda.empty_cache()
            elif index == 3:
                print("2-shot CoT")
                config.llm_client_config.temperature = 0
                config.llm_client_config.prompting_technique = "CoT"
                config.llm_client_config.number_consistency_path = 0
                config.llm_client_config.is_few_shot = True
                config.llm_client_config.ordering_few_shot_examples = "PN"
                config.llm_client_output_directory_path = output_path + "2s_CoT/"
                slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
                run_experiment_with_evaluation(config, client, 0, slr_name)
                torch.cuda.empty_cache()
            elif index == 4:
                print("2-shot CoT (n=3)")
                config.llm_client_config.temperature = 0.5
                config.llm_client_config.prompting_technique = "CoT"
                config.llm_client_config.number_consistency_path = 2
                config.llm_client_config.is_few_shot = True
                config.llm_client_config.ordering_few_shot_examples = "PN"
                config.llm_client_output_directory_path = output_path + "2s_CoT_sc/"
                slr_name = slr_name = os.path.basename(slr).split(".csv")[0]
                run_experiment_with_evaluation(config, client, 0, slr_name)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', type=int, required=True, help='Index of the job')
    args = parser.parse_args()
    main(args.index, False)
