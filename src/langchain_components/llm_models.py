import os
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_llm(config: Dict[str, Any]) -> Optional[BaseChatModel]: # Adjust return type if needed
    """
    Initializes and returns the specified LLM based on the config.
    Supports 'google', 'openai', and 'local' (vLLM) providers.

    Args:
        config: Dictionary containing configuration:
            - LLM_PROVIDER: 'google', 'openai', or 'local'.
            - LLM_MODEL_NAME: Name of the model for the specified provider.
            - LLM_TEMPERATURE: Sampling temperature (might not be supported by all vLLM models/configs).
            - LLM_MAX_TOKENS: Maximum number of tokens to generate.
            - GOOGLE_API_KEY_ENV_VAR: Env var name for Google API key (if provider is 'google').
            - OPENAI_API_KEY_ENV_VAR: Env var name for OpenAI API key (if provider is 'openai').
            - OPENAI_BASE_URL: Base URL for OpenAI API (if provider is 'openai').
            - VLLM_OPENAI_API_BASE: Base URL for the local vLLM OpenAI-compatible server (if provider is 'local').
            # Add other provider-specific config keys as needed

    Returns:
        An instance of the specified LangChain LLM (ChatModel or LLM), or None if initialization fails.
    """
    provider = config.get('LLM_PROVIDER', 'google').lower() # Default to google
    model_name = config.get('LLM_MODEL_NAME')
    temperature = config.get('LLM_TEMPERATURE', 0.0) # Note: Temperature might behave differently or not be supported by VLLMOpenAI depending on the backend model/config
    max_tokens = config.get('LLM_MAX_TOKENS') # Let provider handle default if None
    openai_api_base = config.get('VLLM_OPENAI_API_BASE', "http://localhost:8001/v1") # For local provider

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

        elif provider == 'local':
            # Configuration specific to local vLLM provider
            openai_api_base = config.get('VLLM_OPENAI_API_BASE') # e.g., "http://localhost:8000/v1"
            # model_name is already fetched above

            if not openai_api_base:
                logging.error("VLLM_OPENAI_API_BASE is missing in the configuration for the 'local' provider.")
                return None

            logging.info(f"Initializing VLLMOpenAI model: {model_name} via {openai_api_base}")

            # Parameters for VLLMOpenAI might differ slightly or have specific requirements
            # Check langchain_community.llms.VLLMOpenAI documentation for exact parameters
            llm = VLLMOpenAI(
                openai_api_key="EMPTY", # Typically required, even if empty for local servers
                openai_api_base=openai_api_base,
                model_name=model_name,
                temperature=temperature, # Pass temperature if supported
                max_tokens=max_tokens if max_tokens is not None else -1, # vLLM might use -1 or other value for unlimited
                # model_kwargs could be used for other parameters if needed
                # model_kwargs={"stop": ["\n"]} # Example
            )
            # Note: VLLMOpenAI returns an LLM, not a ChatModel.
            # You might need to adjust how you use it or adjust the function's return type.
            logging.info("VLLMOpenAI model initialized successfully.")
            # IMPORTANT: VLLMOpenAI is likely an LLM, not a BaseChatModel.
            # If your downstream code strictly expects BaseChatModel, this will cause issues.
            # You might need to adjust the return type annotation and how the returned object is used.
            return llm # Returning the LLM instance

        else:
            logging.error(f"Unsupported LLM provider specified: {provider}")
            return None

    except Exception as e:
        logging.error(f"Failed to initialize LLM provider '{provider}' with model '{model_name}': {e}", exc_info=True)
        return None