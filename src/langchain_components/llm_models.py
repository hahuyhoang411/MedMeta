import os
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_llm(config: Dict[str, Any]) -> Optional[BaseChatModel]:
    """
    Initializes and returns the specified Chat LLM based on the config.
    Supports 'google' and 'openai' providers.

    Args:
        config: Dictionary containing configuration:
            - LLM_PROVIDER: 'google' or 'openai'.
            - LLM_MODEL_NAME: Name of the model for the specified provider.
            - LLM_TEMPERATURE: Sampling temperature.
            - LLM_MAX_TOKENS: Maximum number of tokens to generate.
            - GOOGLE_API_KEY_ENV_VAR: Env var name for Google API key (if provider is 'google').
            - OPENAI_API_KEY_ENV_VAR: Env var name for OpenAI API key (if provider is 'openai').
            # Add other provider-specific config keys as needed (e.g., OPENAI_BASE_URL)

    Returns:
        An instance of the specified BaseChatModel, or None if initialization fails.
    """
    provider = config.get('LLM_PROVIDER', 'google').lower() # Default to google
    model_name = config.get('LLM_MODEL_NAME')
    temperature = config.get('LLM_TEMPERATURE', 0.0)
    # Max tokens might need provider-specific defaults or handling
    max_tokens = config.get('LLM_MAX_TOKENS') # Let provider handle default if None

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
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else 32000,
                google_api_key=api_key,
                # convert_system_message_to_human=True # May be needed depending on model/usage
            )
            logging.info("ChatGoogleGenerativeAI model initialized successfully.")
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

        else:
            logging.error(f"Unsupported LLM provider specified: {provider}")
            return None

    except Exception as e:
        logging.error(f"Failed to initialize LLM provider '{provider}' with model '{model_name}': {e}", exc_info=True)
        return None