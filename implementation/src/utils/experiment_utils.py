import sys
import os
import time
import copy
from typing import Tuple
sys.path.append(os.getcwd())
from implementation.src.data.guo_data_provider import GuoDataProvider
from implementation.src.data.tar_data_provider import TarDataProvider
from implementation.src.utils.data_utils import exclude_few_shot_examples_from_dataframe, get_ids_of_few_shot_examples
import pandas as pd
from implementation.src.config.config_loader import ConfigLoader
from implementation.src.config.config import Config
from implementation.src.prompts.prompt_builder import PromptBuilder
from implementation.src.utils.file_utils import generate_foldername, load_json_file
from implementation.src.prompts.prompt_templates import CoT_prompt_template
from implementation.src.prompts.prompt_handler import PromptHandler
from implementation.src.data.results_handler import ResultHandler
from implementation.src.utils.file_utils import save_to_json
from implementation.src.data.synergy_data_provider import SynergyDataProvider
from implementation.src.prompts.prompt_templates import zero_shot_prompt_template, CoT_prompt_template, zero_shot_prompt_template_binary
from implementation.src.prompts.system_message_templates import system_message_rq, system_message_basic
from implementation.src.utils.data_utils import initialize_few_shot_examples
from implementation.src.client.baseline_client import BaselineClient

CONFIG_PATH = "config.json"

def create_few_shot_examples(client, index: int, slr_name: str):
    """
    Creates few shot examples.

    :param client: LLM client.
    :param index: Index of the cross-validation split.
    :param slr_name: Name of slr.
    """
    slr_name = slr_name.split(".csv")[0]
    # load and validate config
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.config
    few_shot_folder = config.folder_path_few_shot_examples
    backup_n_c_paths = copy.deepcopy(config.llm_client_config.number_consistency_path)
    backup_is_few_s = copy.deepcopy(config.llm_client_config.is_few_shot)
    backup_few_s_order = copy.deepcopy(config.llm_client_config.ordering_few_shot_examples)
    config.llm_client_config.is_few_shot = False
    config.llm_client_config.ordering_few_shot_examples = ""
    config.llm_client_config.number_consistency_path = 0
    
    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    # load the dataset (potential relevant papers 
    prompt_builder = PromptBuilder(config)

    # load the dataset (potential relevant papers for the SLR)
    slr_df = data_provider.create_dataframe(f"{slr_name}.csv")

    # load metadata of SLR
    slr_infos_df = load_json_file(config.file_path_slr_infos)[slr_name]

    prompt_template = []
    prompting_type = config.llm_client_config.prompting_technique
    if prompting_type == "CoT":
        prompt_template.append(CoT_prompt_template)
    elif int(config.relevance_upper_value) - int(config.relevance_lower_value) == 1:
        prompt_template.append(zero_shot_prompt_template_binary)
    else:
        prompt_template.append(zero_shot_prompt_template)
    system_message = system_message_rq
    backup_temp = copy.deepcopy(config.llm_client_config.temperature)
    backup_sm = copy.deepcopy(config.llm_client_config.system_message_type)
    backup_output_dir = copy.deepcopy(config.llm_client_output_directory_path)
    config.llm_client_config.temperature = 0
    config.llm_client_config.system_message_type = system_message_rq
    config.llm_client_output_directory_path = few_shot_folder + "llm_logs/"

    prompt_handler = PromptHandler(
        config=config,
        client=client,
        prompt_builder=prompt_builder,
        prompt_template=prompt_template,
        slr_infos_df=slr_infos_df,
        dataset=slr_df,
        system_message=system_message,
        few_shot_examples=None,
        slr_name=slr_name
    )
    print("Sending prompts to LLM ...", end="\n")
    folder_name, file_names = prompt_handler.evaluate_papers_by_llm_client(index)

    folder_path = config.llm_client_output_directory_path + folder_name
    # initiate result handler
    result_handler = ResultHandler(
        config=config,
        folder_path=folder_path
    )
    file_path = folder_path + file_names[0]
    result_df, _, _ = result_handler.process_json_to_dataframe(file_path)

    # merge dataset dataframe with result dataframe
    slr_df["id"] = slr_df["id"].apply(str)
    merged_df = pd.merge(slr_df, result_df, on="id", validate="one_to_one")
    merged_df = merged_df[["id", "title", "abstract", "label", "question", "response", "decision_of_llm", "relevance_of_paper"]]
    merged_df = merged_df[merged_df["relevance_of_paper"] != -1]
    optimal_relevant_papers = merged_df[merged_df["label"] == 1]
    optimal_relevant_papers = optimal_relevant_papers.loc[optimal_relevant_papers.index.isin(merged_df[merged_df["relevance_of_paper"] == int(config.relevance_upper_value)].index)]

    optimal_irrelevant_papers = merged_df[merged_df["label"] == 0]
    optimal_irrelevant_papers = optimal_irrelevant_papers.loc[optimal_irrelevant_papers.index.isin(merged_df[merged_df["relevance_of_paper"] == int(config.relevance_lower_value)].index)]

    optimal_df = pd.concat([optimal_relevant_papers, optimal_irrelevant_papers])
    if prompting_type == "CoT":
        optimal_df = optimal_df.loc[optimal_df["response"].str.contains("Let's think step by step")]
    optimal_df = optimal_df.to_dict("records")
    save_to_json(optimal_df, f"{slr_name}_{prompting_type}_ex_{config.llm_client_config.name_of_model}_point_{index}.json", few_shot_folder)
    print(f"Successfully created {len(optimal_df)} few shot examples")
    # restore old config properties
    config.llm_client_config.number_consistency_path = backup_n_c_paths
    config.llm_client_config.temperature = backup_temp
    config.llm_client_config.system_message_type = backup_sm
    config.llm_client_output_directory_path = backup_output_dir
    config.llm_client_config.is_few_shot = backup_is_few_s
    config.llm_client_config.ordering_few_shot_examples = backup_few_s_order

def run_experiment_with_evaluation(config: Config, client, index: int, slr_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs an experiment with evaluation for the given index of the cross-validation split.

    :param config: Configuration object.
    :param client: LLM client.
    :param index: Index of the cross-validation split.
    :param slr_name: Name of slr.
    """
    # initialization
    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    prompt_builder = PromptBuilder(config)

    # load the dataset (potential relevant papers for the SLR)
    slr_df = data_provider.create_dataframe(f"{slr_name}.csv")
    few_shot_examples = initialize_few_shot_examples(config, index, slr_name)

    # exclude few shot examples from dataset for evaluation
    slr_df = exclude_few_shot_examples_from_dataframe(
        ids_list=get_ids_of_few_shot_examples([example for sublist in few_shot_examples for example in sublist]), dataframe=slr_df
    )

    # load metadata of SLR
    slr_infos_df = load_json_file(config.file_path_slr_infos)[slr_name]

   # get prompting technique (either default prompting or CoT)
    prompting_type = config.llm_client_config.prompting_technique
    print("Creating prompt handler ...", end="\n")

    # set the prompt template based on the prompting technique
    prompt_template = []
    if prompting_type == "CoT":
        prompt_template.append(CoT_prompt_template)
    elif int(config.relevance_upper_value) - int(config.relevance_lower_value) == 1:
        prompt_template.append(zero_shot_prompt_template_binary)
    else:
        prompt_template.append(zero_shot_prompt_template)
    
    system_message = None
    if config.llm_client_config.system_message_type == "system_message_rq":
        system_message = system_message_rq
    else:
        system_message = system_message_basic

    prompt_handler = PromptHandler(
        config=config,
        client=client,
        prompt_builder=prompt_builder,
        prompt_template=prompt_template,
        slr_infos_df=slr_infos_df,
        dataset=slr_df,
        system_message=system_message,
        few_shot_examples=few_shot_examples,
        slr_name=slr_name
    )
    # analyse papers of dataset with llm
    print("Sending prompts to LLM ...", end="\n")
    folder_name, _ = prompt_handler.evaluate_papers_by_llm_client(index)
    folder_path = config.llm_client_output_directory_path + folder_name
    print("Successfully created results JSON of experiment: " + folder_path, end="\n")


def run_experiment_bert_0_shot(config: Config, client: BaselineClient, label: str, index: int, slr_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs an experiment with 0-shot non-fine-tuned LMs for the given index of the cross-validation split.

    :param config: Configuration object.
    :param client: LLM client.
    :param label: Label of the experiment.
    :param index: Index of the cross-validation split.
    :param slr_name: Name of slr.
    """
    # load metadata of SLR
    slr_infos_df = load_json_file(config.file_path_slr_infos)[slr_name]

    query = slr_infos_df["title"]
    if config.llm_client_config.system_message_type == "system_message_rq":
        print("Using title and research questions as query ...")
        research_questions = slr_infos_df["research_questions"].replace("\n", "")
        research_questions = research_questions.replace("- ", "")
        query = slr_infos_df["title"] + " " + research_questions
    else:
        print("Using only title as query ...")

    data_provider = SynergyDataProvider(config)
    if "tar2019" in config.folder_path_slrs:
        data_provider = TarDataProvider(config)
    elif "guo" in config.folder_path_slrs:
        data_provider = GuoDataProvider(config)
    slr_df = data_provider.create_dataframe(f"{slr_name}.csv")

    documents = [str(row["title"]) + " " + str(row["abstract"]) for _, row in slr_df.iterrows()]
    doc_ids = slr_df["id"].apply(str).to_list()

    start_time = time.time()

    relevance_scores = client.get_relevance_scores(query, documents, batch_size=32)

    result_ids = client.get_ranked_ids(relevance_scores, doc_ids)
    elapsed_time = time.time() - start_time
    result_df = pd.DataFrame({"id": result_ids})

    folder_path = config.llm_client_output_directory_path + generate_foldername(label, len(result_df), slr_name, False, index=index)
    print("Successfully created results JSON of experiment: " + folder_path, end="\n")
    result_handler = ResultHandler(
        config=config,
        folder_path=folder_path,
    )
    result_handler.store_results_bert_eval_only(result_df=result_df, computation_time=elapsed_time, query=query, model_name=label, model_path=client.model_path, slr_name=slr_name)
