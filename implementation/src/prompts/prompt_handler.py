import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import datetime
import time
from typing import Dict, List, Tuple
from implementation.src.config.config import Config
from implementation.src.prompts.prompt_builder import PromptBuilder
from implementation.src.client.local_client import LocalClient
from vllm import SamplingParams
from implementation.src.utils.file_utils import (
    generate_foldername,
    save_to_json,
)

class PromptHandler:
    def __init__(self, 
        config: Config, 
        slr_infos_df: pd.DataFrame,
        client: LocalClient,
        dataset: pd.DataFrame,
        prompt_builder: PromptBuilder,
        prompt_template: List[str],
        system_message: str,
        few_shot_examples: Dict,
        slr_name: str
    ) -> None:
        """
        Initialize the PromptHandler class with configuration and required instances.

        :param config: Configuration object.
        :param slr_infos_df: DataFrame containing metadata of SLR.
        :param client: Client instance.
        :param dataset: DataFrame containing the dataset for analysis.
        :param prompt_builder: PromptBuilder instance.
        :param prompt_template: Template string for prompts.
        :param system_message: System message for prompts.
        :param few_shot_examples: Few-shot examples.
        :param slr_name: Name of slr.
        """
        self.config = config
        self.client = client
        self.slr_infos_df = slr_infos_df
        self.dataset = dataset
        self.prompt_builder = prompt_builder
        self.prompt_template = prompt_template
        self.name_of_model = self.config.llm_client_config.name_of_model
        self.model_id = self.name_of_model

        self.is_few_shot = self.config.llm_client_config.is_few_shot
        self.count = self.config.llm_client_config.count
        self.system_message = prompt_builder.create_system_message_prompt(system_message, self.slr_infos_df)
        self.client_temperature = self.config.llm_client_config.temperature
        self.prompting_technique = self.config.llm_client_config.prompting_technique
        self.few_shot_example_seed = self.config.llm_client_config.few_shot_example_seed
        self.slr_tag = slr_name
        self.llm_client_output_directory_path = self.config.llm_client_output_directory_path
        self.number_consistency_path = config.llm_client_config.number_consistency_path
        self.ordering_few_shot_examples = self.config.llm_client_config.ordering_few_shot_examples
        self.num_of_max_requests = self.config.llm_client_config.num_of_max_requests

        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.few_shot_examples = few_shot_examples
        self.start_time = 0
        self.responses = self._initialize_responses()


    def _initialize_responses(self) -> Dict:
        """
        Initialize the responses dictionary with metadata.

        :return: A dictionary initialized with metadata.
        """
        if not isinstance(self.count, int):
            self.count = len(self.dataset)

        responses = {
            "Date and Time": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "deployment_name": self.model_id,
            "path_to_model": self.config.llm_client_config.path_to_model,
            "path_of_additional_ranker": self.config.llm_client_config.path_to_reranker,
            "model_temperature": self.client_temperature,
            "count": self.count,
            "prompt_template": self.prompt_template,
            "system_message": self.system_message,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
            "total_computation_time": 0,
            "num_of_max_requests": self.num_of_max_requests
        }

        if self.is_few_shot:
            responses["ordering_few_shot_examples"] = self.ordering_few_shot_examples
            responses["different_examples"] = self.config.llm_client_config.different_examples
            responses["few_shot_example_seed"] = self.few_shot_example_seed
            if len(self.few_shot_examples) > 0:
                responses["few_shot_examples"] = self.few_shot_examples

        responses["papers"] = {}
        responses["failed_responses"] = {}

        return responses

    def send_prompt_to_llm(self, message_list: List[Dict[str, str]]):
        """
        Send a prompt to the LLM and track the response time.

        :param message_list: List of message dictionaries to send as a prompt.
        :return: Response from the LLM.
        """
        try:
            sampling_params = SamplingParams(max_tokens=1024, temperature=0.5, top_p=1) # temperature=0.5 because it is for failed requests that might need higher temperature
            return self.client.chat(messages=message_list, sampling_params=sampling_params)
        except Exception as e:
            return {"error": str(e)}
        
    def send_prompts_to_llm(self, messages: List[List[Dict[str, str]]]):
        """
        Send a batch of prompts to the LLM using the transformers library and track the response time for each.

        :param message_batch: List of message lists (each list corresponds to a single prompt) to send as a prompt batch.
        :return: Response of the LLM
        """
        try:
            sampling_params = SamplingParams(max_tokens=1024, temperature=self.config.llm_client_config.temperature, top_p=1)
            return self.client.chat(messages=messages, sampling_params=sampling_params)
        except Exception as e:
            return [{"error": str(e)} for _ in messages]

        
    def extract_response_data(self, response: Dict) -> Tuple[str, int, int]:
        """
        Extract response data from the LLM response.

        :param response: Response dictionary from the LLM.
        :return: Extracted response text, prompt tokens, and completion tokens.
        """
        try:
            response_text, response_prompt_tokens, response_completion_tokens = "", "", ""
            response_text = response.outputs[0].text
            response_prompt_tokens = len(response.prompt_token_ids)
            response_completion_tokens = len(response.outputs[0].token_ids)
        except Exception as e:
            print(f"Error: {e}, response: {response}")
            response_text = "An unexpected error occurred."
            response_prompt_tokens = 0
            response_completion_tokens = 0
        return response_text, response_prompt_tokens, response_completion_tokens
    
    def save_response(
        self,
        row: pd.Series,
        message_list: str,
        response_text: str,
        relevance_of_paper: str,
        threshold_value: float,
        response_prompt_tokens: int,
        response_completion_tokens: int,
    ) -> None:
        """
        Save the response data to the responses dictionary.

        :param row: A row from the dataset containing paper details.
        :param message_list: List of message dictionaries sent as a prompt.
        :param response_text: Text of the response from the LLM.
        :param relevance_of_paper: Relevance of paper according to LLM.
        :param threshold_value: Threshold value extracted from the response.
        :param response_prompt_tokens: Number of tokens in the prompt.
        :param response_completion_tokens: Number of tokens in the response.
        """
        self.responses["papers"][row["id"]] = {
            "question": message_list[-1]["content"],
            "response": response_text,
            "decision_of_llm": relevance_of_paper,
            "relevance_of_paper": threshold_value,
            "ground_truth": row["label"],
            "prompt_tokens": response_prompt_tokens,
            "response_tokens": response_completion_tokens,
        }

    def save_response_failed(
        self,
        row: pd.Series,
        message_list: str,
        response_text: str,
        relevance_of_paper: str,
        threshold_value: float,
        response_prompt_tokens: int,
        response_completion_tokens: int,
        try_number: int
    ) -> None:
        """
        Save the response data to the responses dictionary for failed requests.

        :param row: A row from the dataset containing paper details.
        :param message_list: List of message dictionaries sent as a prompt.
        :param response_text: Text of the response from the LLM.
        :param relevance_of_paper: Relevance of paper according to LLM.
        :param threshold_value: Threshold value extracted from the response.
        :param response_prompt_tokens: Number of tokens in the prompt.
        :param response_completion_tokens: Number of tokens in the response.
        :param try_number: The number of the try.
        """
        self.responses["failed_responses"][str(row["id"]) + "_" + str(try_number)] = {
            "question": message_list[-1]["content"],
            "response": response_text,
            "decision_of_llm": relevance_of_paper,
            "relevance_of_paper": threshold_value,
            "prompt_tokens": response_prompt_tokens,
            "response_tokens": response_completion_tokens,
        }

    def update_token_counts(
        self, response_prompt_tokens: int, response_completion_tokens: int
    ) -> None:
        """
        Update the total counts of prompt and response tokens.

        :param response_prompt_tokens: Number of tokens in the prompt.
        :param response_completion_tokens: Number of tokens in the response.
        """
        self.total_prompt_tokens += response_prompt_tokens
        self.total_response_tokens += response_completion_tokens

    def save_final_results(self, res_number: int, foldername: str) -> str:
        """
        Save the final results of the analysis to a JSON file.

        :param res_number: Number of the run (important for self-consistency).
        :param foldername: Name of the folder where the results are stored.
        :return: file name of result file
        """
        
        output_dir = self.llm_client_output_directory_path + foldername
        res_number = str(res_number)
        file_name = f"log_file_{res_number}.json"
        save_to_json(
            data=self.responses,
            filename=file_name,
            output_dir=output_dir,
        )
        return file_name

    def evaluate_papers_by_llm_client(self, index) -> tuple[str, List[str]]:
        """
        Evaluate the papers of the dataset using the LLM client. This method processes a dataset of titles and abstracts by constructing prompts, sending them to the LLM client, and saving the responses.

        :param index: Index of the run.
        :return: Folderpath of the results folder and a list of finalnames (one name for each run).
        """
        count = self.count if isinstance(self.count, int) else len(self.dataset)

        foldername = generate_foldername(
            slr_tag=self.slr_tag,
            name_of_model=self.name_of_model,
            is_few_shot=self.is_few_shot,
            length=count,
            index=index,
        )
        file_names = []

        for i in range(self.number_consistency_path + 1):
            print(f"Starting run {i + 1} ...")
            self.total_prompt_tokens = 0
            self.total_response_tokens = 0
            messages = []
            row_batch = []
            few_shot_example = self.few_shot_examples[i] if self.few_shot_examples else None
            for idx, (index, row) in enumerate(self.dataset.head(count).iterrows(), start=1):
                prompt_message_list = self.prompt_builder.build_prompt_and_message_list(
                    row=row,
                    prompt_template=self.prompt_template[0],
                    slr_context=self.slr_infos_df,
                    system_message=self.system_message,
                    few_shot_examples=few_shot_example
                )
                messages.append(prompt_message_list)
                row_batch.append(row)
            self.start_time = time.time()
            responses = self.send_prompts_to_llm(messages)

            for response, row, message_list in zip(responses, row_batch, messages):
                for j in range(self.num_of_max_requests):
                    response_text, response_prompt_tokens, response_completion_tokens = self.extract_response_data(response)

                    relevance_of_paper = self.prompt_builder.extract_decision(response_text)
                    threshold_value = self.prompt_builder.extract_threshold_value(relevance_of_paper)
                    self.update_token_counts(response_prompt_tokens, response_completion_tokens)
                    if threshold_value != -1 or j == self.num_of_max_requests - 1:
                        self.save_response(
                            row=row,
                            message_list=message_list,
                            response_text=response_text,
                            relevance_of_paper=relevance_of_paper,
                            threshold_value=threshold_value,
                            response_prompt_tokens=response_prompt_tokens,
                            response_completion_tokens=response_completion_tokens,
                        )
                        break
                    else:
                        self.save_response_failed(
                            row=row,
                            message_list=message_list,
                            response_text=response_text,
                            relevance_of_paper=relevance_of_paper,
                            threshold_value=threshold_value,
                            response_prompt_tokens=response_prompt_tokens,
                            response_completion_tokens=response_completion_tokens,
                            try_number=j
                        )
                        # Resend the prompt if it failed according to num_of_max_requests
                        response = self.send_prompt_to_llm(message_list)[0]

            final_filename = self.save_final_results(res_number=i, foldername=foldername)
            file_names.append(final_filename)
            self.responses["total_computation_time"] = time.time() - self.start_time

        return foldername, file_names
