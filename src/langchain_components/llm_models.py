import os
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_llm(config: Dict[str, Any]) -> Optional[BaseChatModel]:
    """
    Initializes and returns the ChatGoogleGenerativeAI LLM.

    Args:
        config: Dictionary containing configuration:
            - LLM_MODEL_NAME: Name of the Gemini model.
            - LLM_TEMPERATURE: Sampling temperature.
            - LLM_MAX_TOKENS: Maximum number of tokens to generate.
            - GOOGLE_API_KEY_ENV_VAR: Name of the environment variable holding the API key.

    Returns:
        An instance of ChatGoogleGenerativeAI, or None if API key is missing or init fails.
    """
    api_key_env_var = config.get('GOOGLE_API_KEY_ENV_VAR', 'GOOGLE_API_KEY')
    api_key = os.environ.get(api_key_env_var)

    if not api_key:
        logging.error(f"Google API key not found in environment variable '{api_key_env_var}'.")
        # Optionally try to prompt user if running interactively, but better to fail in scripts.
        # import getpass
        # api_key = getpass.getpass(f"Enter your Google AI API key (env var '{api_key_env_var}' not set): ")
        # if not api_key:
        #     logging.error("API key not provided.")
        #     return None
        # os.environ[api_key_env_var] = api_key # Set it for the current session
        return None


    model_name = config.get('LLM_MODEL_NAME', 'gemini-2.5-flash-preview-04-17')
    temperature = config.get('LLM_TEMPERATURE', 0.0)
    max_tokens = config.get('LLM_MAX_TOKENS', 32000)

    try:
        logging.info(f"Initializing ChatGoogleGenerativeAI model: {model_name}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=api_key,
            # Add other parameters like top_p, top_k if needed
             convert_system_message_to_human=True # Good practice for some models
        )
        logging.info("ChatGoogleGenerativeAI model initialized successfully.")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}", exc_info=True)
        return None