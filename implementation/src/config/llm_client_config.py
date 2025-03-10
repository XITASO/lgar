import sys
import os
sys.path.append(os.getcwd())
from pydantic import BaseModel, Field
from typing import Optional, Literal


class LLMClientConfig(BaseModel):
    is_few_shot: bool = Field(
        ...,
        description="Determines whether few-shot examples will be added to the prompts.",
        examples=[True],
    )
    few_shot_example_seed: int = Field(
        ..., description="Seed used for randomly choosing few-shot examples.", examples=[42]
    )
    prompting_technique: Literal["zero_shot", "CoT", "few_shot"] = Field(
        ..., description="Prompting technique to use.", examples=["CoT"]
    )
    temperature: float = Field(
        ..., ge=0, le=2, description="Temperature setting for the LLM client.", examples=[1]
    )
    count: Optional[int] = Field(
        None,
        description="Optional parameter specifying the number of items to be analyzed. If omitted, the entire dataset will be iterated and evaluated by the LLM.",
        examples=[10],
    )
    number_consistency_path: int = Field(
        ..., ge=0, description="Number of times the run will be repeated for self-consistency", examples=[4]
    )
    system_message_type: Literal["system_message_rq", "system_message_basic"] = Field(
        ..., description="Type of the system message that should be used", examples=["system_message_rq"]
    )
    ordering_few_shot_examples : Literal["", "PN", "NP", "PPNN", "NNPP", "PNPN", "NPNP", "PNNP", "NPPN"] = Field(
        ..., description="Ordering that should be used for few shot examples", examples=["PN"]
    )
    num_of_max_requests: int = Field(
        ..., ge=0, description="Maximum number the same request is sent (if the response was not in the desired format)", examples=[4]
    )
    name_of_model: Literal["Llama3.1-8B", "Llama3.1-70B", "Llama3.3-70B", "ColBERT", "SciBERT", "monoT5", "monoBERT", "Qwen2.5-72B", "Qwen2.5-32B", "Mistral"] = Field(
        ..., description="Name of model which is used in current experiment", examples=["Llama3.1-70B"]
    )
    path_to_model: str = Field(
        ..., description="Path to the model which is used in current experiment (only necessary for local models)", examples=["./path/to/Meta-Llama-3.1-8B-Instruct"]
    )
    path_to_reranker: str = Field(
        ..., description="Path to reranker model or checkpoint for downloading it (if possible)", examples=["colbert-ir/colbertv2.0"]
    )
    different_examples: bool = Field(
        ..., description="Determines whether different examples should be used for each self-consistency run (should only be set to true, if we have multiple runs)", examples=[True]
    )
