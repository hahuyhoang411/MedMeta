import os
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
# Remove VLLMOpenAI import as it's no longer used
# from langchain_community.llms import VLLMOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_llm(config: Dict[str, Any]) -> Optional[BaseChatModel]:
    """
    Initializes and returns the specified LLM based on the config.
    Supports 'google', 'openai', and 'local' providers.
    'local' provider uses ChatOpenAI configured for a local OpenAI-compatible endpoint.

    Args:
        config: Dictionary containing configuration:
            - LLM_PROVIDER: 'google', 'openai', or 'local'.
            - LLM_MODEL_NAME: Name of the model for the specified provider.
            - LLM_TEMPERATURE: Sampling temperature.
            - LLM_MAX_TOKENS: Maximum number of tokens to generate.
            - GOOGLE_API_KEY_ENV_VAR: Env var name for Google API key (if provider is 'google').
            - OPENAI_API_KEY_ENV_VAR: Env var name for OpenAI API key (if provider is 'openai').
            - OPENAI_BASE_URL: Base URL for OpenAI API (if provider is 'openai').
            - VLLM_OPENAI_API_BASE: Base URL for the local OpenAI-compatible server (if provider is 'local').
            - OPENAI_MAX_RETRIES: Max retries for OpenAI calls (used for 'openai' and 'local').
            - OPENAI_TIMEOUT: Timeout for OpenAI calls (used for 'openai' and 'local').
            - LLM_GOOGLE_THINKING_BUDGET: Thinking budget for Google Generative AI (if provider is 'google').
            # Add other provider-specific config keys as needed

    Returns:
        An instance of the specified LangChain BaseChatModel, or None if initialization fails.
    """
    provider = config.get('LLM_PROVIDER', 'google').lower() # Default to google
    model_name = config.get('LLM_MODEL_NAME')
    temperature = config.get('LLM_TEMPERATURE', 0.0)
    max_tokens = config.get('LLM_MAX_TOKENS') # Let provider handle default if None
    # Use VLLM_OPENAI_API_BASE for the local provider's base_url
    openai_api_base = config.get('VLLM_OPENAI_API_BASE', "http://localhost:8001/v1") # Default for local
    google_thinking_budget = config.get('LLM_GOOGLE_THINKING_BUDGET')

    logging.info(f"Attempting to initialize LLM provider: {provider}")

    if not model_name:
        logging.error("LLM_MODEL_NAME is missing in the configuration.")
        return None

    try:
        if provider == 'google':
            api_key_env_var = config.get('GOOGLE_API_KEY_ENV_VAR', 'GOOGLE_API_KEY')
            api_key = os.environ.get(api_key_env_var)
            if not api_key:
                logging.error(f"Google API key not found in environment variable '{api_key_env_var}'.")
                return None

            logging.info(f"Initializing ChatGoogleGenerativeAI model: {model_name}")
            llm_params = {
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens if max_tokens is not None else 32000,
                "google_api_key": api_key,
            }
            if google_thinking_budget is not None:
                llm_params["thinking_budget"] = google_thinking_budget

            llm = ChatGoogleGenerativeAI(**llm_params)
            logging.info("ChatGoogleGenerativeAI model initialized successfully.")
            return llm

        if provider == 'vertex-google':
            api_key_env_var = config.get('GOOGLE_API_KEY_ENV_VAR', 'GOOGLE_API_KEY')
            api_key = os.environ.get(api_key_env_var)
            if not api_key:
                logging.error(f"Google API key not found in environment variable '{api_key_env_var}'.")
                return None

            logging.info(f"Initializing ChatGoogleGenerativeAI model: {model_name}")
            llm = ChatVertexAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else 32000,
                google_api_key=api_key,
            )
            logging.info("ChatVertexAI model initialized successfully.")
            return llm
        
        elif provider == 'openai':
            api_key_env_var = config.get('OPENAI_API_KEY_ENV_VAR', 'OPENAI_API_KEY')
            api_key = os.environ.get(api_key_env_var)
            if not api_key:
                logging.error(f"OpenAI API key not found in environment variable '{api_key_env_var}'.")
                return None

            # Optional: Add other OpenAI specific params from config if needed
            base_url = config.get('OPENAI_BASE_URL')
            organization = config.get('OPENAI_ORGANIZATION')
            max_retries = config.get('OPENAI_MAX_RETRIES', 2)
            timeout = config.get('OPENAI_TIMEOUT')


            logging.info(f"Initializing ChatOpenAI model: {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else 32000,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                max_retries=max_retries,
                timeout=timeout,
                # other params...
            )
            logging.info("ChatOpenAI model initialized successfully.")
            return llm

        elif provider == 'local':
            # Use ChatOpenAI for local provider, pointing to the specified base URL
            local_api_base = config.get('VLLM_OPENAI_API_BASE') # Get the specific config for local base URL

            if not local_api_base:
                logging.error("VLLM_OPENAI_API_BASE is missing in the configuration for the 'local' provider.")
                return None

            # Use a placeholder API key, as local endpoints often don't require one
            local_api_key = "NA" # Or "EMPTY" or config.get('LOCAL_API_KEY', 'NA') if you want it configurable

            # Get other relevant OpenAI parameters from config or use defaults
            max_retries = config.get('OPENAI_MAX_RETRIES', 2)
            timeout = config.get('OPENAI_TIMEOUT')

            logging.info(f"Initializing ChatOpenAI model for 'local' provider: {model_name} via {local_api_base}")
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else 32000, # Or adjust default as needed
                api_key=local_api_key,
                base_url=local_api_base,
                max_retries=max_retries,
                timeout=timeout,
                # other params if needed...
            )
            logging.info("ChatOpenAI model for 'local' provider initialized successfully.")
            return llm # ChatOpenAI is a BaseChatModel

        else:
            logging.error(f"Unsupported LLM provider specified: {provider}")
            return None

    except Exception as e:
        logging.error(f"Failed to initialize LLM provider '{provider}' with model '{model_name}': {e}", exc_info=True)
        return None