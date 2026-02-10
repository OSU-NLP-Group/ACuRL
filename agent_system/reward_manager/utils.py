from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)
import os
import backoff

class vllm_OpenaiEngine():
    def __init__(self, model, base_url):
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)


    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=lambda details: print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")
    )
    def generate(self, messages, max_new_tokens=4096, temperature=0, **kwargs):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return [choice.message.content for choice in response.choices]
        

class OpenaiEngine():
    def __init__(
        self,
        rate_limit=-1,
        model=None,
        **kwargs,
    ) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.client = OpenAI(api_key=self.api_key)

        self.model = model  # Save the model parameter
        self.reasoning_models = ["o4-mini","gpt-5","gpt-5-mini"]
        
    def log_error(self, details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=lambda details: print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")
    )
    def generate(self, messages, max_new_tokens=4096, temperature=0, **kwargs):
        if self.model in self.reasoning_models:
            response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            **kwargs,
        )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        return [choice.message.content for choice in response.choices]