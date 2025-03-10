import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
from typing import Dict
from implementation.src.config.config import Config
import re
from typing import Union, List
import string

class PromptBuilder:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.file_path_slr_infos = self.config.file_path_slr_infos
        self.is_few_shot = self.config.llm_client_config.is_few_shot
        self.prompting_technique = self.config.llm_client_config.prompting_technique

        self.seed = self.config.llm_client_config.few_shot_example_seed
        self.relevance_lower_value = config.relevance_lower_value
        self.relevance_upper_value = config.relevance_upper_value

    def extract_decision(self, input_string: str) -> str:
        """
        Extract decision value from a string.

        :param input_string: Input string containing the decision value.
        :return: Extracted decision value.
        """
        pattern = r"Decision: (\d+)"
        match = re.search(pattern, input_string)
        return match.group(0) if match else ""

    def extract_threshold_value(self, string: str) -> Union[float, str]:
        """
        Extract threshold value from a string.

        :param string: Input string containing the threshold value.
        :return: Extracted threshold value.
        """
        lower_value = int(self.relevance_lower_value)
        upper_value = int(self.relevance_upper_value)

        pattern = rf"Decision: (\d+)"
        match = re.search(pattern, string)
        if match:
            value = int(match.group(1))
            if lower_value <= value <= upper_value:
                return value
        return -1.0

    
    def create_prompt(
        self,
        paper: Dict,
        prompt_template: str,
        slr_context: Dict
    ) -> str:
        """
        Create a formatted prompt using provided SLR context and given template

        :param paper: Dictionary with title and abstract of the paper that should be analyzed by LLM.
        :param prompt_template: String template for creating the prompt.
        :param slr_context: Dictionary with context of the SLR (e.g. inclusion/exclusion criteria, ...)
        :return: Formatted prompt string.
        """
        formatted_prompt = prompt_template.format(
            title_paper=paper["title"], 
            abstract = paper["abstract"],
            inclusion_criteria = slr_context["inclusion_criteria"],
            exclusion_criteria = slr_context["exclusion_criteria"],
            relevance_lower_value = self.relevance_lower_value,
            relevance_upper_value = self.relevance_upper_value
            )
        return formatted_prompt
    
    def create_system_message_prompt(self, system_message: str, slr_context: Dict) -> str:
        """
        Create a system message prompt using provided SLR context and message template.

        :param system_message: Template of the system message.
        :param slr_context: Dictionary with context of the SLR (e.g. inclusion/exclusion criteria, ...)
        :return: Formatted system message string.
        """
        context_fields = [k[1] for k in string.Formatter().parse(format_string=system_message) if k[1] is not None]

        values_of_context_fields = [(key, slr_context[key]) for key in context_fields if key in slr_context]
        formatted_system_message = system_message.format_map(dict(values_of_context_fields))
        return formatted_system_message
    
    def build_prompt_and_message_list(
        self, row: pd.Series, prompt_template: str, slr_context: Dict, system_message: str, few_shot_examples: List
    ) -> list[dict]:
        """
        Build a prompt and message list for zero shot or few shot prompting.

        :param row: A row from the dataset containing paper details.
        :param prompt_template: Template for the prompt.
        :param slr_context: Dictionary with context of the SLR (e.g. inclusion/exclusion criteria, ...).
        :param system_message: Systemmessage for LLM.
        :param few_shot_examples: Few shot examples that should be used for this run.
        :return: A list of message dictionaries for the prompt.
        """
        prompt = self.create_prompt(
            paper=row,
            prompt_template=prompt_template,
            slr_context=slr_context,
        )
        message_list = [{"role": "system", "content": system_message}]

        if self.is_few_shot:
            for example in few_shot_examples:
                message_list.extend(
                    [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["response"]},
                    ]
                )

        message_list.append({"role": "user", "content": prompt})
        return message_list
