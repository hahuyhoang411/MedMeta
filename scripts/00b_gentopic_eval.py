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
# Set level to logging.DEBUG for more verbose output (shows each classification attempt)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. Load the data ---
try:
    df = pd.read_csv("meta_analysis_summary.csv")
    logging.info("CSV file loaded successfully.")
    logging.info(df.info())
except FileNotFoundError:
    logging.error("Error: meta_analysis_summary.csv not found. Please make sure the file is in the correct directory.")
    exit() # Exit if the file isn't found

base_prompt_template="""You are a highly specialized AI designed for classifying biomedical literature based on its subject matter.

Your task is to analyze a given PubMed article title and assign it to the *single most relevant category* from the predefined list provided below.

Here is the complete and exclusive list of valid categories you can choose from. These are based on a subset of MeSH (Medical Subject Headings):

```
MESH Categories:
* 		Anatomy
    * 		Body Regions [A01]
    * 		Musculoskeletal System [A02]
    * 		Digestive System [A03]
    * 		Respiratory System [A04]
    * 		Urogenital System [A05]
    * 		Endocrine System [A06]
    * 		Cardiovascular System [A07]
    * 		Nervous System [A08]
    * 		Sense Organs [A09]
    * 		Tissues [A10]
    * 		Cells [A11]
    * 		Fluids and Secretions [A12]
    * 		Animal Structures [A13]
    * 		Stomatognathic System [A14]
    * 		Hemic and Immune Systems [A15]
    * 		Embryonic Structures [A16]
    * 		Integumentary System [A17]
    * 		Plant Structures [A18]
    * 		Fungal Structures [A19]
    * 		Bacterial Structures [A20]
    * 		Viral Structures [A21]
* 		Diseases
    * 		Infections [C01]
    * 		Neoplasms [C04]
    * 		Musculoskeletal Diseases [C05]
    * 		Digestive System Diseases [C06]
    * 		Stomatognathic Diseases [C07]
    * 		Respiratory Tract Diseases [C08]
    * 		Otorhinolaryngologic Diseases [C09]
    * 		Nervous System Diseases [C10]
    * 		Eye Diseases [C11]
    * 		Urogenital Diseases [C12]
    * 		Cardiovascular Diseases [C14]
    * 		Hemic and Lymphatic Diseases [C15]
    * 		Congenital, Hereditary, and Neonatal Diseases and Abnormalities [C16]
    * 		Skin and Connective Tissue Diseases [C17]
    * 		Nutritional and Metabolic Diseases [C18]
    * 		Endocrine System Diseases [C19]
    * 		Immune System Diseases [C20]
    * 		Disorders of Environmental Origin [C21]
    * 		Animal Diseases [C22]
    * 		Pathological Conditions, Signs and Symptoms [C23]
    * 		Occupational Diseases [C24]
    * 		Chemically-Induced Disorders [C25]
    * 		Wounds and Injuries [C26]
* 		Chemicals and Drugs
    * 		Inorganic Chemicals [D01]
    * 		Organic Chemicals [D02]
    * 		Heterocyclic Compounds [D03]
    * 		Polycyclic Compounds [D04]
    * 		Macromolecular Substances [D05]
    * 		Hormones, Hormone Substitutes, and Hormone Antagonists [D06]
    * 		Enzymes and Coenzymes [D08]
    * 		Carbohydrates [D09]
    * 		Lipids [D10]
    * 		Amino Acids, Peptides, and Proteins [D12]
    * 		Nucleic Acids, Nucleotides, and Nucleosides [D13]
    * 		Complex Mixtures [D20]
    * 		Biological Factors [D23]
    * 		Biomedical and Dental Materials [D25]
    * 		Pharmaceutical Preparations [D26]
    * 		Chemical Actions and Uses [D27]
* 		Analytical, Diagnostic and Therapeutic Techniques, and Equipment
    * 		Diagnosis [E01]
    * 		Therapeutics [E02]
    * 		Anesthesia and Analgesia [E03]
    * 		Surgical Procedures, Operative [E04]
    * 		Investigative Techniques [E05]
    * 		Dentistry [E06]
    * 		Equipment and Supplies [E07]
* 		Psychiatry and Psychology
    * 		Behavior and Behavior Mechanisms [F01]
    * 		Psychological Phenomena [F02]
    * 		Mental Disorders [F03]
    * 		Behavioral Disciplines and Activities [F04]
```

**Classification Rules:**
1.  Read the provided PubMed article title carefully.
2.  Determine the primary subject or focus of the article based on its title.
3.  Select the *single* category from the list above that best matches the title's primary subject. Aim for the most specific category provided in the list that applies.
4.  **You MUST NOT create any new categories or choose any category not explicitly listed above.**
5.  If the article title's subject matter does not clearly and appropriately fit into *any* of the categories provided in the list, you must classify it as **"Other"**.

**Output Format:**
*   Your response must consist of **ONLY** the name of the selected category.
*   **Do not** include the MESH code (e.g., do not output "[A01]").
*   **Do not** include any introductory phrases, explanations, or additional text.
*   If you classify the article as "Other", your output should be the single word: `Other`.

---
**Example Input (User provides this):**
Cardiovascular risk factors in patients with type 2 diabetes mellitus

**Example Output (AI should respond like this):**
Cardiovascular Diseases

---
**Example Input (User provides this):**
Genetic analysis of *Escherichia coli* virulence factors

**Example Output (AI should respond like this):**
Bacterial Structures

---
**Now, classify the following PubMed Article Title:**

{article_title}""" # Placeholder for the title from the DataFrame

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

# --- 5. Worker function for a single task ---
def classify_title_task(title, base_prompt, model_name, client, max_retries):
    """
    Task function to classify a single title.
    Includes prompt formatting, API call, retry logic, and result cleaning.
    Takes title as the first argument, followed by fixed arguments.
    Returns the classification string or "Classification Error".
    """
    current_prompt = base_prompt.format(article_title=title)
    classification = "Classification Error" # Default error state

    for attempt in range(max_retries):
        try:
            # Use the model instance to generate content
            response = client.models.generate_content(
                model=model_name,
                contents=current_prompt
            )

            # Extract and clean the response text
            classification = response.text.strip()

            # Check if the response seems empty or invalid
            if not classification:
                 raise ValueError("Received empty or whitespace-only classification")

            logging.debug(f"Successfully classified '{title}' as '{classification}' on attempt {attempt + 1}")
            return classification # Success, return the classification

        except Exception as e:
            # Log a warning for the specific title and attempt
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for title '{title[:50]}...': {e}") # Truncate long titles for logging
            if attempt < max_retries - 1:
                wait_time = 2 * (2 ** attempt) # Exponential backoff
                logging.warning(f"Waiting {wait_time} seconds before retrying title '{title[:50]}...'...")
                time.sleep(wait_time)
            else:
                logging.error(f"Max retries reached for title '{title[:50]}...'. Assigning '{classification}'.")
                # classification is still "Classification Error" here

    return classification # Return the default error state if all retries fail


# --- 6. Process Titles using Multithreading ---

# Determine which part of the DataFrame to process based on DEBUG_MODE
if DEBUG_MODE:
    df_to_process = df.head(DEBUG_ROWS).copy() # Use .copy() to avoid SettingWithCopyWarning later
    total_rows_to_process = DEBUG_ROWS
    logging.info(f"DEBUG MODE: Processing only the first {DEBUG_ROWS} rows.")
else:
    df_to_process = df.copy() # Work on a copy to avoid potential issues with shared data access
    total_rows_to_process = len(df)
    logging.info(f"Starting multithreaded AI classification for {total_rows_to_process} titles with {MAX_WORKERS} workers...")

# Extract the titles to process into a list
titles_to_process = df_to_process['Meta Analysis Name'].tolist()

# Create a partial function with fixed arguments bound
# This function now only needs the 'title' argument when called by thread_map
partial_classify_task = partial(
    classify_title_task,
    base_prompt=base_prompt_template,
    model_name=model_name,
    client=client,
    max_retries=max_retries
)

# Use thread_map from tqdm.contrib.concurrent for easy multithreading with a progress bar
# Pass the partial function and the list of titles.
# fn_kwargs is no longer needed here because the fixed args are in the partial function.
topic_results = thread_map(
    partial_classify_task,          # The partial function to execute
    titles_to_process,              # The iterable of arguments (each title)
    max_workers=MAX_WORKERS,        # Set the number of concurrent threads
    desc="Classifying Titles (Multithreaded)" # Progress bar description
)


logging.info("Multithreaded processing complete.")

# --- 7. Add the Results as a New Column ---
# The 'topic_results' list returned by thread_map is in the same order as 'titles_to_process',
# which corresponds to the order of rows in df_to_process.
# We assign these results back to the original DataFrame using the index from df_to_process.
df.loc[df_to_process.index, 'Topic'] = topic_results


# --- 8. Display Results (Optional) ---
print("\nProcessing complete.")
print("\nDataFrame with new 'Topic' column:")
# Ensure 'Category' column exists before trying to display it, or handle missing column
cols_to_display = ['Meta Analysis Name', 'Topic']
if 'Category' in df.columns:
    cols_to_display.append('Category')
print(df[cols_to_display].head())

print("\nUpdated DataFrame Info:")
print(df.info())

# --- 9. Save the Updated DataFrame ---
output_filename = "meta_analysis_summary_with_topics.csv"
df.to_csv(output_filename, index=False) # index=False prevents writing the pandas index as a column
print(f"\nUpdated data saved to {output_filename}")