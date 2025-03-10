import sys
import os
sys.path.append(os.getcwd())
from implementation.src.config.config import Config
import torch
from vllm import LLM, SamplingParams

class LocalClient():
    def __init__(self, config: Config) -> None:
        """
        Initialize the LocalClient class with a configuration object.

        :param config: Configuration object.
        """
        self.config = config
        self.client_temp = self.config.llm_client_config.temperature
        self.path_to_model = self.config.llm_client_config.path_to_model
        print(f"torch count: {torch.cuda.device_count()}")
        self.client = LLM(model=self.path_to_model, tokenizer=self.path_to_model, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95, max_model_len=20000, enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=1024)

    def send_test_prompt(self) -> None:
        """
        Send a test prompt to the LLM client and print the result.
        """
        prompt = [{"role": "user", "content": "What model am I talking to?"}]

        response = self.client.chat(messages=prompt, sampling_params=SamplingParams(max_tokens=256))
        if not response or not response[0].outputs:
            raise ValueError("Failed to receive a valid response from the LLM client.")
        print(f"Prompt: {prompt}")
        print(f'Response: {response[0].outputs[0].text}')
