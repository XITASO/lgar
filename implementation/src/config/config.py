import sys
import os
sys.path.append(os.getcwd())
from pydantic import BaseModel, Field
from implementation.src.config.llm_client_config import LLMClientConfig


class Config(BaseModel):
    folder_path_slrs: str = Field(
        ...,
        description="Path to folder with the csv files of the given slr dataset",
        examples=["./implementation/data/guo_slrs/lpvr.csv"],
    )
    file_path_slr_infos: str = Field(
        ...,
        description="Path to the json file with the metadata of the SLRs (inclusion/exclusion criteria, title, ...)",
        examples=["./implementation/data/guo_slrs/guo_infos.json"],
    )
    llm_client_output_directory_path: str = Field(
        ...,
        description="Directory path for LLM client output.",
        examples=["./data/experiment_results/"],
    )
    llm_client_config: LLMClientConfig = Field(
        ..., description="Configuration settings for the LLM client."
    )
    relevance_lower_value: str = Field(
        ..., description="Lower value of the scale (for relevance score of LLM)", examples=["0"],
    )
    relevance_upper_value: str = Field(
        ..., description="Upper value of the scale (for relevance score of LLM)", examples=["4"],
    )
    folder_path_few_shot_examples: str = Field(
        ...,
        description="Path to folder with the .json files for few-shot examples",
        examples=["./data/few_shot_examples/"],
    )
