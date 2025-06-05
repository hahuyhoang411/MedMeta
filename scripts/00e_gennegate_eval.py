import pandas as pd
from google import genai # Use the standard alias
import getpass
import time # Import time for potential delays (helpful for API limits)
import logging # Import the logging module
from tqdm import tqdm # Import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed # For managing threads manually
from tqdm.contrib.concurrent import thread_map # A simpler way with tqdm integration
from functools import partial # Import partial

# --- 0. Configuration and Setup (Add DEBUG_MODE and Logging) ---
DEBUG_MODE = False  # Set to True to process only the first 5 rows
DEBUG_ROWS = 10    # Number of rows to process in debug mode
MAX_WORKERS = 3   # Number of concurrent threads for API calls

# Configure logging
# Set level to logging.DEBUG for more verbose output (shows each negation attempt)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. Load the data ---
try:
    df = pd.read_csv("../pubmed_data_reference_eval.csv")
    logging.info("CSV file loaded successfully.")
    logging.info(df.info())
except FileNotFoundError:
    logging.error("Error: pubmed_data_reference_eval.csv not found. Please make sure the file is in the correct directory.")
    exit() # Exit if the file isn't found


# --- 2. Define prompt template for negation ---
base_prompt_template = """You are a medical research assistant. Your task is to create a negated/opposite version of the given meta-analysis text while maintaining scientific credibility and plausibility.

Given the following meta-analysis title and abstract, create a similar text but with conclusions that are opposite or contradictory to the original. Make sure to:
1. Keep the same title format and structure
2. Maintain the same study design and methodology description
3. Change only the findings/conclusions to be opposite or contradictory
4. Ensure the negated conclusion is medically plausible and realistic
5. Use appropriate medical terminology and maintain scientific rigor

Original text:
{original_text}

Create a negated version with opposite conclusions:"""

# --- 3. Configure the Google AI Client ---
# The client is generally thread-safe for concurrent calls.
try:
    api_key = getpass.getpass("Enter your Google API key: ")
    client = genai.Client(api_key=api_key)
except Exception as e:
    logging.error(f"Failed to initialize Google AI Client: {e}")
    exit()

# --- 4. Initialize the Model ---
model_name = "gemini-2.5-flash-preview-04-17"
max_retries = 5 # Maximum retries for API calls within a worker

# --- 5. Worker function for a single negation task ---
def negate_text_task(original_text, base_prompt, model_name, client, max_retries):
    """
    Task function to negate a single meta-analysis text.
    Includes prompt formatting, API call, retry logic, and result cleaning.
    Takes original_text as the first argument, followed by fixed arguments.
    Returns the negated text string or "Negation Error".
    """
    current_prompt = base_prompt.format(original_text=original_text)
    negated_text = "Negation Error" # Default error state

    for attempt in range(max_retries):
        try:
            # Use the model instance to generate content
            response = client.models.generate_content(
                model=model_name,
                contents=current_prompt
            )

            # Extract and clean the response text
            negated_text = response.text.strip()

            # Check if the response seems empty or invalid
            if not negated_text:
                 raise ValueError("Received empty or whitespace-only negated text")

            logging.debug(f"Successfully negated text (first 50 chars): '{original_text[:50]}...' on attempt {attempt + 1}")
            return negated_text # Success, return the negated text

        except Exception as e:
            # Log a warning for the specific text and attempt
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for text (first 50 chars): '{original_text[:50]}...': {e}")
            if attempt < max_retries - 1:
                wait_time = 2 * (2 ** attempt) # Exponential backoff
                logging.warning(f"Waiting {wait_time} seconds before retrying text (first 50 chars): '{original_text[:50]}...'...")
                time.sleep(wait_time)
            else:
                logging.error(f"Max retries reached for text (first 50 chars): '{original_text[:50]}...'. Assigning '{negated_text}'.")
                # negated_text is still "Negation Error" here

    return negated_text # Return the default error state if all retries fail


# --- 6. Process Texts using Multithreading ---

# Determine which part of the DataFrame to process based on DEBUG_MODE
if DEBUG_MODE:
    df_to_process = df.head(DEBUG_ROWS).copy() # Use .copy() to avoid SettingWithCopyWarning later
    total_rows_to_process = DEBUG_ROWS
    logging.info(f"DEBUG MODE: Processing only the first {DEBUG_ROWS} rows.")
else:
    df_to_process = df.copy() # Work on a copy to avoid potential issues with shared data access
    total_rows_to_process = len(df)
    logging.info(f"Starting multithreaded text negation for {total_rows_to_process} texts with {MAX_WORKERS} workers...")

# Extract the texts to process into a list
texts_to_process = df_to_process['text'].tolist()

# Create a partial function with fixed arguments bound
# This function now only needs the 'original_text' argument when called by thread_map
partial_negate_task = partial(
    negate_text_task,
    base_prompt=base_prompt_template,
    model_name=model_name,
    client=client,
    max_retries=max_retries
)

# Use thread_map from tqdm.contrib.concurrent for easy multithreading with a progress bar
# Pass the partial function and the list of texts.
negation_results = thread_map(
    partial_negate_task,          # The partial function to execute
    texts_to_process,              # The iterable of arguments (each text)
    max_workers=MAX_WORKERS,        # Set the number of concurrent threads
    desc="Negating Meta-Analysis Texts (Multithreaded)" # Progress bar description
)


logging.info("Multithreaded processing complete.")

# --- 7. Create New DataFrame with Negated Results ---
# Create a new DataFrame with the same structure but with negated texts
df_negated = df_to_process.copy()
df_negated['text'] = negation_results

# --- 8. Display Results (Optional) ---
print("\nNegation processing complete.")
print("\nOriginal vs Negated samples:")
for i in range(min(3, len(df_negated))):  # Show first 3 examples
    print(f"\n--- Example {i+1} ---")
    print(f"ORIGINAL: {df['text'].iloc[i][:200]}...")
    print(f"NEGATED:  {df_negated['text'].iloc[i][:200]}...")

print(f"\nTotal rows processed: {len(df_negated)}")

# --- 9. Save the Negated DataFrame ---
output_filename = "../pubmed_data_reference_eval_negate.csv"
df_negated.to_csv(output_filename, index=False) # index=False prevents writing the pandas index as a column
print(f"\nNegated data saved to {output_filename}")
logging.info(f"Negated dataset saved to {output_filename} with {len(df_negated)} rows")