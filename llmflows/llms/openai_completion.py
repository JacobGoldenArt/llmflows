# pylint: disable=too-few-public-methods, R0913, W0221

"""
This module implements a wrapper for OpenAI chat completion models for gpt-3.5-turbo and up, using BaseLLM as a
base class. Use this module for completions only not general chat. For general chat use openai_chat.py.
"""

import openai
from openai.error import (
    APIError,
    Timeout,
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
)
from .llm import BaseLLM
from .llm_utils import call_with_retry, async_call_with_retry


class OpenAICompletion(BaseLLM):
    """
    A class for interacting with the OpenAI chat completion API.

    Inherits from BaseCompletionLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI API.

    Args:
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        api_key (str): The API key to use for interacting with the OpenAI API.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.5,
        max_tokens: int = 400,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        super().__init__(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.verbose = verbose
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide OpenAI API key")
        openai.api_key = self._api_key

    def prepare_results(self, model_outputs, retries) -> tuple[str, dict, dict]:
        """
        Formats results after generation.

        Args:
            model_outputs: Raw output after model generation.
            retries (int): Number of retries taken for successful generation.

        Returns:
            A tuple containing the generated text, the raw response data, and the
                model configuration.
        """
        text_result = model_outputs.choices[0]["message"]

        call_data = {
            "raw_outputs": model_outputs,
            "retries": retries,
        }

        model_config = {
            "model_name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        return text_result, call_data, model_config

    def generate(self, prompt: str) -> tuple[str, dict, dict]:
        """
         Generates a Chat Completion from the user prompt using OpenAI API.

        Args:
            prompt: A string representing the prompt to generate text from.

         Returns:
             A tuple containing the generated text, the raw response data, and the
                 model configuration.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        # transform prompt string to a list for chat completion api.
        message = [{"role": "user", "content": prompt}]

        completion, retries = call_with_retry(
            func=openai.ChatCompletion.create,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
            ),
            max_retries=self.max_retries,
            model=self.model,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self.prepare_results(completion, retries)

    async def generate_async(self, prompt: str) -> tuple[str, dict, dict]:
        """
        Generates a text completion using OpenAI API asynchronously.

           Args:
               prompt: A string representing the prompt to generate text from.


        Returns:
            A tuple containing the generated text, the raw response data, and the
                model configuration.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        # transform prompt string to array for chat completion api.
        message = [
            {"role": "user", "content": prompt},
        ]

        completion, retries = await async_call_with_retry(
            async_func=openai.ChatCompletion.acreate,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
            ),
            max_retries=self.max_retries,
            model=self.model,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self.prepare_results(completion, retries)
