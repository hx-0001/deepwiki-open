import logging
from typing import Dict, Any, Optional
import aiohttp
import requests
from api.config import ZHUGESHENMA_API_KEY
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType

log = logging.getLogger(__name__)

class ZhugeShencodeClient(ModelClient):
    """A client for ZhugeShenma (诸葛神码) API chat completions."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or ZHUGESHENMA_API_KEY or ''
        self.base_url = base_url or 'http://shenma.sangfor.com:9080/v1'
        self.sync_client = self.init_sync_client()
        self.async_client = None

    def init_sync_client(self):
        return {
            "api_key": self.api_key,
            "base_url": self.base_url
        }

    def init_async_client(self):
        return {
            "api_key": self.api_key,
            "base_url": self.base_url
        }

    def convert_inputs_to_api_kwargs(self, input: Any, model_kwargs: Dict = None, model_type: ModelType = None) -> Dict:
        model_kwargs = model_kwargs or {}
        if model_type == ModelType.LLM:
            messages = []
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, list) and all(isinstance(msg, dict) for msg in input):
                messages = input
            else:
                raise ValueError(f"Unsupported input format for ZhugeShenma: {type(input)}")
            api_kwargs = {
                "model": model_kwargs.get("model", "deepseek-v3"),
                "messages": messages,
                "temperature": model_kwargs.get("temperature", 0.7),
                "stream": model_kwargs.get("stream", False),
                "max_tokens": model_kwargs.get("max_tokens", 2000)
            }
            return api_kwargs
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/RooVetGit/Roo-Cline",
            "X-Title": "Roo Code"
        }

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.LLM):
        if model_type != ModelType.LLM:
            raise ValueError("Only LLM chat completion is supported.")
        url = f"{self.base_url}/chat/completions"
        headers = self.get_headers()
        try:
            response = requests.post(url, headers=headers, json=api_kwargs, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"ZhugeShenma API call failed: {e}")
            raise

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.LLM):
        if model_type != ModelType.LLM:
            raise ValueError("Only LLM chat completion is supported.")
        url = f"{self.base_url}/chat/completions"
        headers = self.get_headers()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=api_kwargs, timeout=60) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            log.error(f"ZhugeShenma async API call failed: {e}")
            raise 