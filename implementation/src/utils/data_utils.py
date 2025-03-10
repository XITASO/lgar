import traceback
from typing import List, Dict
import pandas as pd
import random
import bm25s
from implementation.src.client.monoBERT_client import MonoBERTClient
from implementation.src.client.monoT5_client import MonoT5Client
from implementation.src.config.config import Config
from implementation.src.utils.file_utils import load_json_file

def create_ranking_pointwise(result_df: pd.DataFrame, slr_infos_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Creates a list of articles that is ranked by the relevance of the papers.

    :param result_df: Result dataframe (unsorted).
    :param slr_infos_df: Dataframe with metadata of SLR.
    :param config: Configuration object.
    :return: Dataframe that is sorted by relevance (descending).
    """
    result_df["relevance_of_paper"] = result_df["relevance_of_paper"].astype(float)
    result_df.loc[result_df["relevance_of_paper"] == -1, "relevance_of_paper"] = result_df["relevance_of_paper"].mean()
    result_df = result_df.sort_values(by="relevance_of_paper", ascending=False, kind="mergesort") # use stable sort
    result_df_grouped = result_df.groupby("relevance_of_paper")
    unique_vals = result_df["relevance_of_paper"].unique()
    new_df = pd.DataFrame()
    query = slr_infos_df["title"]
    if config.llm_client_config.system_message_type == "system_message_rq":
        print("Using title and research questions as query.")
        research_questions = slr_infos_df["research_questions"].replace("\n", "")
        research_questions = research_questions.replace("- ", "")
        query = slr_infos_df["title"] + " " + research_questions
    else:
        print("Using only title as query.")
    for val in unique_vals:
        new_group = result_df_grouped.get_group(val)
        if new_group.shape[0] > 1:
            try:
                client = None
                path_to_reranker = config.llm_client_config.path_to_reranker
                if "monoBERT" in path_to_reranker:
                    client = MonoBERTClient(path_to_reranker)
                elif "monoT5" in path_to_reranker:
                    client = MonoT5Client(path_to_reranker)
                documents = [str(row["title"]) + " " + str(row["abstract"]) for _, row in new_group.iterrows()]
                doc_ids = new_group["id"].apply(str).to_list()
                scores = client.get_relevance_scores(query=query, documents=documents, batch_size=32)
                ranked_ids = client.get_ranked_ids(scores, doc_ids)
                result_df = pd.DataFrame({"id": ranked_ids})
                result_df["id"] = pd.to_numeric(result_df["id"])
                result_df = pd.merge(result_df, new_group, how="inner", on="id")
                new_df = pd.concat([new_df, result_df])
                print(f"used {path_to_reranker} for re-ranking group")
            except Exception as e:
                print(f"Error: {e}, provided path {path_to_reranker}")
                print("No valid path for re-ranker provided; BM25 will be used as fallback.")
                traceback.print_exc()
                new_df = pd.concat([new_df, create_bm_25_ranking(new_group, query=query)])
        else:
            new_df = pd.concat([new_df, new_group])
    return new_df

def get_ids_of_few_shot_examples(few_shot_examples: Dict) -> List[int]:
    """
    Returns the ids of all few shot examples in the Dict

    :param few_shot_examples: Few shot examples.
    :param config: Configuration object.
    :return: List of the ids of all few shot examples
    """
    return [int(example["id"]) for example in few_shot_examples]

def exclude_few_shot_examples_from_dataframe(
    ids_list: List[int], dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Exclude rows from a DataFrame based on a list of ids.

    :param ids_list: A list of strings containing ids to be excluded.
    :param dataframe: A pandas DataFrame containing data to be filtered.
    :return: A new DataFrame with rows excluded where the 'id' column value matches any string in the ids_list.
    """
    ids_set = set(ids_list)
    return dataframe[~dataframe["id"].isin(ids_set)]

def get_few_shot_examples(config: Config, index: int, slr_name) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads potential few shot examples and selects randomly the amount of wanted examples.
    
    :param config: The Configuration of the run.
    :param index: Index of the run for the few shot examples.
    :param slr_name: SLR name.
    :return: Dataframe of relevant examples and Dataframe of irrelevant examples
    """
    few_shot_examples = load_json_file(config.folder_path_few_shot_examples + slr_name + "_" + config.llm_client_config.prompting_technique + "_ex_"+ config.llm_client_config.name_of_model + f"_point_{index}.json")
    few_shot_examples = pd.json_normalize(few_shot_examples)
    relevant = few_shot_examples[few_shot_examples["label"] == 1].to_dict("records")
    irrelevant = few_shot_examples[few_shot_examples["label"] == 0].to_dict("records")
    random.seed(config.llm_client_config.few_shot_example_seed) # set the seed for reproducibility

    # same amount of positive and negative examples
    num_examples = len(config.llm_client_config.ordering_few_shot_examples) // 2
    if config.llm_client_config.different_examples:
        num_examples *= (config.llm_client_config.number_consistency_path + 1)
    relevant_examples, irrelevant_examples = pd.DataFrame, pd.DataFrame
    if len(relevant) > 0:
        relevant_examples = random.choices(relevant, k=num_examples)
    if len (irrelevant) > 0:
        irrelevant_examples = random.choices(irrelevant, k=num_examples)
    return relevant_examples, irrelevant_examples

def initialize_few_shot_examples(config: Config, index: int, slr_name: str) -> List[List]:
    """
    Initializes the array with few shot examples if it is a few shot run.

    :param config: The Configuration of the run.
    :param index: Index of the run for the few shot examples.
    :param slr_name: Name of slr.
    :return: List of few shot examples
    """
    few_shot_examples = []
    if config.llm_client_config.is_few_shot:
        if len(config.llm_client_config.ordering_few_shot_examples) > 0:
            relevant_examples, irrelevant_examples = None, None
            relevant_examples, irrelevant_examples = get_few_shot_examples(config, index, slr_name)
            if not isinstance(relevant_examples, list):
                relevant_examples = []
            if not isinstance(irrelevant_examples, list):
                irrelevant_examples = []
            if not config.llm_client_config.different_examples and config.llm_client_config.number_consistency_path > 0:
                relevant_examples = relevant_examples * (config.llm_client_config.number_consistency_path + 1)
                irrelevant_examples = irrelevant_examples * (config.llm_client_config.number_consistency_path + 1)
            for i in range(config.llm_client_config.number_consistency_path + 1):
                relevant_count, irrelevant_count = 0, 0
                few_shot_example_run = []
                for x in config.llm_client_config.ordering_few_shot_examples:
                    if x == "P" and relevant_count < len(relevant_examples):
                        few_shot_example_run.append(relevant_examples[relevant_count + i * len(relevant_examples) // (config.llm_client_config.number_consistency_path + 1)])
                        relevant_count += 1
                    elif x == "N" and irrelevant_count < len(irrelevant_examples):
                        few_shot_example_run.append(irrelevant_examples[irrelevant_count + i * len(irrelevant_examples) // (config.llm_client_config.number_consistency_path + 1)])
                        irrelevant_count += 1
                few_shot_examples.append(few_shot_example_run)
        else:
            print("Configuration should be changed because you want to have a few shot run, but specified a few shot pattern which uses 0 few shot examples.")
    return few_shot_examples

def create_bm_25_ranking(dataframe: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Creates a dataframe of the given papers which is ranked by relevance to the query by using BM-25.

    :param dataframe: Datframe of papers.
    :param query: Query to which the relevance of each paper is determined.
    :return: Dataframe of papers in ranked order (most relevant papers first).
    """
    abstracts = dataframe["abstract"].astype(str).to_list()
    ids = dataframe["id"].astype(str).tolist()
    corpus_json = []
    for i in range(len(dataframe)):
        corpus_json.append({"text": abstracts[i], "id": ids[i]})
    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(corpus_text)
    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens)

    # Query the corpus
    results, _ = retriever.retrieve(bm25s.tokenize(query), k=len(dataframe))
    results = pd.DataFrame(results[0].tolist())
    results["id"] = results["id"].astype(int)

    # create result dataframe
    result_df = pd.merge(results, dataframe, on="id", validate="one_to_one")
    return result_df
