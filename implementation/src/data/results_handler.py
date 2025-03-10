import datetime
from io import StringIO
import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from typing import List, Tuple
from implementation.src.config.config import Config
from implementation.src.utils.file_utils import ensure_directory_exists, load_json_file, save_to_json

class ResultHandler:
    def __init__(self, config: Config, folder_path: str) -> None:
        """
        Initialize the ResultHandler Class with configuration, JSON file path to the created results by the LLM, and the dataframe of the corresponding SLR dataset.

        :param config: Configuration object.
        :param folder_path: Path of the result folder of this experiment.
        """
        self.config = config
        self.folder_path = folder_path

    def process_json_to_dataframe(self, json_file_path: str) -> Tuple[pd.DataFrame, float]:
        """
        Reads a JSON file, extracts papers and date information, and converts them to a DataFrame.

        :param json_file_path: Path to the JSON result file.
        :return: DataFrame containing LLM responses for all papers and date information as well as percentage of failed requests.
        """

        json_data = load_json_file(json_file_path)

        papers_list = [
            {**value, "id": key} for key, value in json_data["papers"].items()
        ]

        llm_df = pd.json_normalize(papers_list)
        llm_df["Date and Time"] = json_data["Date and Time"]

        perc_failed = len(json_data["failed_responses"]) if len(json_data["papers"]) == 0 else len(json_data["failed_responses"]) / len(json_data["papers"])
        return llm_df, perc_failed, json_data["path_of_additional_ranker"]
    
    def process_ranked_df_json_to_df(self, json_file_path: str) -> Tuple[pd.DataFrame, float]:
        """
        Reads a JSON file, extracts sorted list of papers, and converts them to a DataFrame.

        :param json_file_path: Path to the JSON result file.
        :return: DataFrame of ranked papers.
        """
        json_data = load_json_file(json_file_path)
        ranked_list = pd.read_json(StringIO(json_data["ranked_df"]), orient="records")
        return ranked_list

    def store_mean_metrics_and_std(self, metrics: pd.Series, std: pd.Series, tag: str, with_std: bool) -> None:
        """
        Store mean metrics and standard deviation.

        :param metrics: Metrics that have been calculated and should be stored in json file.
        :param std: Standard deviation of metrics that have been calculated and should be stored in json file.
        :param tag: Tag of slr.
        :param with_std: Indicates whether there is a standard deviation to store.
        """
        ensure_directory_exists(self.folder_path)
        combined_metrics = metrics
        if with_std:
            combined_metrics = metrics.combine(std, lambda mean, std: f"{mean:.5f}±{std:.3f}")
        else:
            combined_metrics = metrics.apply(lambda mean: f"{mean:.5f}")
        combined_metrics.to_json(self.folder_path + f"/{tag}_metrics.json")

    def store_results_bert_eval_only(self, result_df: pd.DataFrame, computation_time: float, query: str, model_name: str, model_path: str, slr_name: str) -> None:
        """
        Stores the results of a ColBERT evaluation as json file.

        :param result_df: Dataframe of ranked ids of ColBERT model.
        :param computation_time: Time it took to load model, to encode the documents, and to generate the ranking.
        :param query: Query that was used for the ranking.
        :param model_name: Name of the model that was used for the ranking.
        :param model_path: Path to the model that was used for the ranking.
        :param slr_name: Tag of SLR.
        """
        json_results = {
            "Date and Time": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "deployment_name": model_name,
            "path_to_model": model_path,
            "query": query,
            "total_computation_time": computation_time,
            "slr_name": slr_name
        }
        json_results["ranked_df"] = result_df.to_json(orient="records")
        save_to_json(json_results, "logfile.json", self.folder_path)

    def create_self_consistency_df(self, file_names: List[str]) -> pd.DataFrame:
        """
        Creates the combined result dataframe from multiple runs by averageing the scores

        :param file_names: Names of files of which the average should be calculated
        :return: combined result dataframe
        """
        result_dfs = None
        column_names = []
        re_ranker = None
        for i, name in enumerate(file_names):
            result_df, _, re_ranker = self.process_json_to_dataframe(self.folder_path + name)
            column_name = "rel_" + name
            column_names.append(column_name)
            result_df = result_df[['id', 'relevance_of_paper']]
            result_df = result_df.rename(columns={"relevance_of_paper": column_name})
            if i == 0:
                result_dfs = result_df
            else:
                result_dfs = result_dfs.merge(result_df, on="id", validate="one_to_one")
        result_dfs["relevance_of_paper"] = np.mean(result_dfs[column_names].values, axis=1)
        return result_dfs, re_ranker
