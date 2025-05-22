import argparse
import logging
import sys
import os
import time
import re
import getpass
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Ensure the src directory is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Project Modules (assuming config stores the output path of script 03)
from src.config import (
    EVALUATION_RESULTS_CSV_PATH, # Input for this script
    OUTPUT_DIR,
    LLM_CONFIG # Reuse LLM config if suitable, or define new ones
)

# LiteLLM for LLM calls
try:
    from litellm import completion
except ImportError:
    print("LiteLLM not installed. Please install it: pip install litellm")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce litellm verbosity
logging.getLogger("litellm").setLevel(logging.WARNING) # Further reduce litellm verbosity

# --- LLM Judge Configuration ---
# Define multiple judges
# NOTE: Ensure vLLM server is running and accessible at the specified api_base
# NOTE: Update vllm_model_id path if necessary
JUDGE_CONFIGS = [
    # {
    #     "name": "Gemini-Flash",
    #     "model": "gemini/gemini-2.5-flash-preview-04-17", # Use latest flash
    #     "api_key_env_var": "GEMINI_API_KEY",
    #     "max_tokens": 8192, # Adjust if needed
    #     "temperature": 0.0,
    #     "api_base": None, # Not needed for Gemini API
    #     "reasoning_effort": "medium"
    # },
    # # Add the vLLM judge based on user's example
    # {
    #     "name": "Local-Qwen2.5-72B",
    #     # Use the exact model ID reported by vLLM's /v1/models endpoint
    #     "model": "hosted_vllm//home/jovyan/visual-thinker-workspace/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31/",
    #     "api_key": "NA", # No API key needed for local vLLM usually
    #     "max_tokens": 8192, # As per user example
    #     "temperature": 0.0, # Keep temperature low for consistency
    #     "api_base": "http://localhost:8000/v1" # Base URL of vLLM server
    # },
    {
        "name": "Gemini-Pro-Low",
        "model": "gemini/gemini-2.5-pro-preview-03-25", # Use latest flash
        "api_key_env_var": "GEMINI_API_KEY",
        "max_tokens": 8192, # Adjust if needed
        "temperature": 0.0,
        "api_base": None, # Not needed for Gemini API
        "reasoning_effort": "low"
    },
    #     {
    #     "name": "Gemini-Pro-Medium",
    #     "model": "gemini/gemini-2.5-pro-preview-03-25", # Use latest flash
    #     "api_key_env_var": "GEMINI_API_KEY",
    #     "max_tokens": 8192, # Adjust if needed
    #     "temperature": 0.0,
    #     "api_base": None, # Not needed for Gemini API
    #     "reasoning_effort": "medium"
    # },
    #     {
    #     "name": "Gemini-Pro-High",
    #     "model": "gemini/gemini-2.5-pro-preview-03-25", # Use latest flash
    #     "api_key_env_var": "GEMINI_API_KEY",
    #     "max_tokens": 8192, # Adjust if needed
    #     "temperature": 0.0,
    #     "api_base": None, # Not needed for Gemini API
    #     "reasoning_effort": "high"
    # },
    {
        "name": "OpenAI-o4-mini-high",
        "model": "o4-mini", # Use latest flash
        "api_key_env_var": "OPENAI_API_KEY",
        "max_tokens": 8192, # Adjust if needed
        "temperature": 1.0, # OpenAi requires temperature to be set at 1.0 for O series
        "api_base": None, # Not needed for OpenRouter API
    },
    # {
    #     "name": "OpenAI-o3",
    #     "model": "o3", # Use latest flash
    #     "api_key_env_var": "OPENAI_API_KEY",
    #     "max_tokens": 8192, # Adjust if needed
    #     "temperature": 1.0, # OpenAi requires temperature to be set at 1.0 for O series
    #     "api_base": None, # Not needed for OpenRouter API
    # },
    {
        "name": "Qwen3-235B-A22B",
        "model": "deepinfra/Qwen/Qwen3-235B-A22B", # Use latest flash
        "api_key_env_var": "DEEPINFRA_API_KEY",
        "max_tokens": 8192, # Adjust if needed
        "temperature": 0.0,
        "api_base": None, # Not needed for OpenRouter API
    }
    # Add more judges here if needed
    # {
    #     "name": "Another-Model",
    #     "model": "provider/model-name",
    #     "api_key_env_var": "OTHER_API_KEY", # Or direct key
    #     "max_tokens": 4096,
    #     "temperature": 0.0,
    #     "api_base": None # Or specify if needed
    # }
]

# Output file for judged results
JUDGED_EVALUATION_RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "medmeta_evaluation_llm_multi_judged.csv") # Updated filename

# --- Helper Functions ---

def get_api_key(env_var="GEMINI_API_KEY"):
    """Gets the API key from environment variables, prompting if necessary."""
    # This function might need further generalization if many keys are needed,
    # but for now, we fetch specific keys based on judge config.
    api_key = os.environ.get(env_var)
    if not api_key:
        logging.warning(f"{env_var} not found in environment variables.")
        try:
            # Only prompt if it's interactive and key is missing
            if sys.stdin.isatty():
                 api_key = getpass.getpass(f"Enter your API key for {env_var}: ")
                 os.environ[env_var] = api_key # Store for the session
            else:
                 logging.error(f"Cannot prompt for {env_var} in non-interactive mode.")
                 return None
        except Exception as e:
            logging.error(f"Could not get API key for {env_var}: {e}")
            return None
    if not api_key:
        logging.error(f"API key ({env_var}) is missing or empty.")
        return None
    return api_key

def extract_score_and_reasoning(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extracts score and reasoning from the LLM judge response."""
    score = None
    reasoning = None

    # Extract Score
    score_match = re.search(r'Score:[\s\*]*(\d+)', text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            logging.warning(f"Could not parse score from match: {score_match.group(1)}")

    # Assign the full text to reasoning.
    # This replaces the previous specific reasoning extraction and the complex fallback logic.
    reasoning = text.strip() if text else "LLM response was empty"

    # Log if score extraction failed.
    if score is None:
        if text: # Check if text is not empty before trying to slice it for the log
            logging.warning(f"Could not extract score from text: '{text[:100]}...'. Full text is used as reasoning.")
        else:
            logging.warning("Could not extract score, and the LLM response was empty.")

    return score, reasoning

# Modify get_llm_judgment to accept judge_config
def get_llm_judgment(row: pd.Series, judge_config: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """
    Sends data to a specific LLM judge and gets the score and reasoning.

    Args:
        row: A pandas Series representing a row from the evaluation results.
        judge_config: A dictionary containing the configuration for the judge LLM.

    Returns:
        A tuple containing the extracted score (int) and reasoning (str).
    """
    original_conclusion = row.get('Conclusion', 'N/A')
    generated_conclusion = row.get('Generated Conclusion', 'N/A')
    judge_name = judge_config.get("name", "Unknown Judge")

    # --- Prepare API Call Parameters ---
    model = judge_config.get("model")
    if not model:
        logging.error(f"[{judge_name}] Model name missing in judge configuration.")
        return None, f"Error: Model name missing for judge {judge_name}"

    api_key = judge_config.get("api_key") # Direct key if provided
    if not api_key and "api_key_env_var" in judge_config:
        # Fetch key from environment if env var name is specified
        api_key = get_api_key(judge_config["api_key_env_var"])
        if not api_key:
             # If key fetch fails, log error and return
             logging.error(f"[{judge_name}] Failed to get API key from {judge_config['api_key_env_var']}")
             return None, f"Error: Missing API key for judge {judge_name}"

    api_base = judge_config.get("api_base")
    max_tokens = judge_config.get("max_tokens", 8192) # Default if not specified
    temperature = judge_config.get("temperature", 0.0) # Default if not specified

    # Define the system prompt (remains the same)
    system_prompt = """## Persona & Objective
You are an expert **Clinical Research Scientist and Critical Appraiser** specializing in meta-analysis methodology and scientific communication. Your objective is to rigorously evaluate the quality and semantic similarity of a conclusion generated by an Agentic RAG system (designed to automate meta-analysis) against the original conclusion from a published, peer-reviewed meta-analysis.

## Input Data
You will receive:
1.  `[Generated Conclusion]`: The text generated by the Agentic RAG system.
2.  `[Original Conclusion]`: The verbatim conclusion text from the published meta-analysis.

## Core Task
Evaluate the `[Generated Conclusion]` based on its semantic alignment and completeness compared to the `[Original Conclusion]`. Assign a similarity score from 0 to 5 using the detailed rubric below. Provide a structured justification for your score, referencing specific aspects of the comparison.

## Evaluation Criteria
Focus on the **semantic meaning and the core components** typically found in meta-analysis conclusions. Evaluate the `[Generated Conclusion]`'s alignment with the `[Original Conclusion]` across these dimensions:

1.  **Main Finding(s) / Overall Result:**
    *   Does it accurately capture the primary outcome(s) or effect(s) reported in the original? (e.g., treatment effectiveness, diagnostic accuracy, association strength, lack of effect).
    *   Does it reflect the direction and magnitude (if specified qualitatively or quantitatively) of the main finding(s)? (e.g., "significantly reduced LDL-C", "did not reduce mortality", "better clinical outcomes").
2.  **Key Specifics & Comparisons:**
    *   Does it mention the specific interventions, populations, or contexts discussed in the original? (e.g., "short-acting beta-blockers in septic patients", "pitavastatin at 1 mg, 2 mg, 4 mg", "ML based on high-resolution CT", "ACLR with ALLR").
    *   Does it include crucial comparisons highlighted in the original? (e.g., dose comparisons, comparison to other treatments/methods like "compared to isolated ACLR", "compared to less commonly used statins").
    *   Does it capture key quantitative results if present and central to the original conclusion? (e.g., "% reduction", specific sensitivity/specificity levels if mentioned).
3.  **Nuance, Caveats, and Limitations:**
    *   Does it reflect any major limitations, caveats, or calls for caution mentioned in the original? (e.g., "high heterogeneity", "interpret with caution", "further research needed", "remains uncertain", need for "multicenter clinical trials").
    *   Does it capture the level of certainty or confidence expressed in the original?
4.  **Implications & Future Directions:**
    *   Does it reflect the key clinical implications, significance, or recommendations for future research stated in the original? (e.g., "provide new insights into targeted treatment", "potential of AI-based tools", "support an individualized approach", "serve as a foundation for incorporation into clinical practice").
5.  **Safety/Tolerability (if applicable):**
    *   If the original conclusion addresses safety, adverse effects, or tolerability, does the generated conclusion accurately reflect this aspect? (e.g., "well tolerated and safe", "low incidence of myalgia", "increases in liver enzymes").
6.  **Overall Semantic Equivalence:**
    *   Considering all the above, does the generated conclusion convey the same core message and essential details as the original, even if the wording differs?

## Scoring Rubric (0-5 Scale)

*   **5: Excellent Similarity / Semantically Equivalent**
    *   Accurately captures all main findings, key specifics, essential nuance/caveats, and core implications/future directions from the original.
    *   Minor differences in wording are acceptable if the meaning is preserved entirely. Conveys the same overall message and takeaway points. Includes safety aspects if mentioned in the original.
    *   Essentially, a reader would draw the exact same conclusions from both texts regarding the study's outcome and significance.

*   **4: High Similarity / Mostly Equivalent**
    *   Accurately captures the main findings and most key specifics.
    *   May miss minor details, some nuance/caveats, or less critical implications OR phrase them slightly differently but without changing the core meaning.
    *   The primary takeaway message is the same.

*   **3: Moderate Similarity / Partially Equivalent**
    *   Captures the main finding(s) correctly but misses significant supporting details, comparisons, nuance, limitations, or implications mentioned in the original.
    *   OR captures most elements but introduces a minor inaccuracy or misrepresentation that slightly alters the emphasis or completeness.
    *   A reader gets the general gist but misses important context or qualifications present in the original.

*   **2: Low Similarity / Superficially Related**
    *   Captures *some* element related to the topic but misrepresents the main finding(s) or omits crucial information necessary to understand the original conclusion's core message.
    *   OR focuses on a minor point from the original while ignoring the central conclusion.
    *   There's a connection, but the essential meaning differs significantly.

*   **1: Very Low Similarity / Barely Related**
    *   Mentions the same general topic but the stated conclusions are substantially different, contradictory in parts, or completely miss the scope and findings of the original.
    *   Fails to capture almost all key evaluation criteria accurately.

*   **0: No Similarity / Contradictory or Irrelevant**
    *   The generated conclusion is on a completely different topic, directly contradicts the main findings of the original, or is nonsensical/irrelevant.

## Instructions for Evaluation

1.  **Read Carefully:** Thoroughly read and understand both the `[Generated Conclusion]` and the `[Original Conclusion]`.
2.  **Compare Systematically:** Evaluate the `[Generated Conclusion]` against the `[Original Conclusion]` using the **Evaluation Criteria** outlined above. Note points of alignment and divergence for each criterion.
3.  **Determine Score:** Based on your systematic comparison, select the score (0-5) from the **Scoring Rubric** that best reflects the overall semantic similarity and completeness.
4.  **Formulate Justification:** Write a concise yet comprehensive justification for your score.
    *   Start by stating the score.
    *   Explicitly reference the **Evaluation Criteria** (e.g., "The generated conclusion accurately captured the main finding regarding X but failed to mention the crucial caveat about heterogeneity...").
    *   Provide specific examples from both texts to support your assessment. Highlight key agreements and disagreements in meaning.
    *   Focus on *semantic content* rather than identical phrasing.

## Output Format

Provide your evaluation in the following structure:

Justification: [Your detailed explanation comparing the generated conclusion to the original conclusion based on the specified criteria and rubric. Explain *why* you assigned the score, referencing specific points of agreement and disagreement in their core message, details, nuance, and implications.]

Score: [Your score from 0-5]
"""

    # Define the user prompt content
    user_content = f"""
### Inputs
#### Generated Conclusion
{generated_conclusion}

#### Original Conclusion
{original_conclusion}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logging.info(f"[{judge_name}] Calling model: {model}" + (f" via {api_base}" if api_base else ""))
    try:
        response = completion(
            model=model,
            messages=messages,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Handle potential differences in response structure if necessary
        if response and response.choices and response.choices[0].message:
             response_text = response.choices[0].message.content
             logging.debug(f"[{judge_name}] Raw Response: {response_text[:200]}...")
             score, reasoning = extract_score_and_reasoning(response_text)
             # Add judge name to reasoning for clarity
             reasoning_with_judge = f"[{judge_name}] {reasoning}"
             return score, reasoning_with_judge
        else:
             logging.error(f"[{judge_name}] Received invalid or empty response object: {response}")
             return None, f"[{judge_name}] Error: Invalid response object"


    except Exception as e:
        logging.error(f"[{judge_name}] Error calling LLM judge for conclusion '{original_conclusion[:50]}...': {e}", exc_info=False) # Set exc_info=False for less verbose logs unless debugging
        # Optionally log traceback if needed: logging.exception(f"...")
        return None, f"[{judge_name}] Error during LLM call: {type(e).__name__}"


# --- Main Function ---

def main(input_file: str, output_file: str, max_rows: Optional[int], wait_time: int):
    """
    Main function to load evaluation results, get LLM judgments from multiple judges,
    calculate mean scores, and save.

    Args:
        input_file: Path to the evaluation results CSV from script 03.
        output_file: Path to save the LLM-judged evaluation results CSV.
        max_rows: Maximum number of rows to process (None for all).
        wait_time: Seconds to wait between LLM calls *for each judge* (for rate limits).
    """
    judge_start_time = time.time()

    # --- Pre-fetch API Keys if needed (optional optimization) ---
    # You could pre-fetch keys here to avoid doing it in the loop,
    # especially if prompting is required. For now, get_api_key handles it.
    logging.info("Pre-fetching API keys if necessary...")
    processed_env_vars = set() # To avoid prompting for the same env var multiple times
    for config in JUDGE_CONFIGS:
        api_key_env = config.get("api_key_env_var")
        if api_key_env and api_key_env not in processed_env_vars:
            # If api_key is not directly provided and not in os.environ yet for this env_var
            if not config.get("api_key") and not os.environ.get(api_key_env):
                logging.info(f"Checking API key for environment variable: {api_key_env}")
                get_api_key(api_key_env) # This will prompt if needed and store in os.environ
            processed_env_vars.add(api_key_env)
    logging.info("Finished pre-fetching API keys.")


    # --- Load Evaluation Data ---
    logging.info(f"Loading evaluation results from: {input_file}")
    if not os.path.exists(input_file):
        logging.error(f"Input evaluation file not found: {input_file}. Exiting.")
        return
    try:
        df_eval = pd.read_csv(input_file)
        logging.info(f"Loaded {len(df_eval)} rows for LLM judgment.")
    except Exception as e:
        logging.error(f"Failed to load evaluation CSV: {e}", exc_info=True)
        return

    # Limit rows if specified
    if max_rows is not None and max_rows > 0:
        logging.warning(f"Processing only the first {max_rows} rows for LLM judgment.")
        df_eval = df_eval.head(max_rows)
    elif max_rows == 0:
         logging.warning("Max rows set to 0. No LLM judgment will be performed.")
         return

    num_rows_to_process = len(df_eval)
    # Store aggregated results per row
    aggregated_scores = []
    # Store individual judge results per row (optional, for detailed analysis)
    individual_judge_results: Dict[str, List[Optional[Any]]] = {f"{cfg['name']}_Score": [] for cfg in JUDGE_CONFIGS}
    individual_judge_results.update({f"{cfg['name']}_Reasoning": [] for cfg in JUDGE_CONFIGS})


    # --- Process Each Row ---
    logging.info(f"Starting LLM judgment loop for {num_rows_to_process} rows using {len(JUDGE_CONFIGS)} judges...")
    for index, row in tqdm(df_eval.iterrows(), total=num_rows_to_process, desc="Judging Rows"):
        start_row_time = time.time()
        original_number = row.get('Number', f'Index_{index}') # For logging

        logging.info(f"\n--- Judging Row {index+1}/{num_rows_to_process} (Number: {original_number}) ---")

        row_scores = [None] * len(JUDGE_CONFIGS) # Initialize with Nones to preserve order
        row_reasonings = [None] * len(JUDGE_CONFIGS) # Initialize with Nones

        # Skip rows where the original conclusion was skipped or errored
        if str(row.get('Generated Conclusion', '')).startswith('Skipped') or \
           str(row.get('Generated Conclusion', '')).startswith('Error'):
            logging.warning(f"Skipping LLM judgment for row {index+1} due to previous skip/error.")
            mean_score = None
            combined_reasoning = "Skipped - Prior Error/Skip"
            for i, cfg in enumerate(JUDGE_CONFIGS):
                 # No need to append to row_scores/row_reasonings here, they remain None
                 individual_judge_results[f"{cfg['name']}_Score"].append(None)
                 individual_judge_results[f"{cfg['name']}_Reasoning"].append("Skipped - Prior Error/Skip")
        else:
            # --- Parallel execution for judges ---
            with ThreadPoolExecutor(max_workers=len(JUDGE_CONFIGS)) as executor:
                future_to_judge_idx = {}
                for judge_idx, judge_config in enumerate(JUDGE_CONFIGS):
                    judge_name = judge_config.get("name", f"Judge_{judge_idx}")
                    logging.info(f"--- Submitting task for Judge {judge_idx+1}/{len(JUDGE_CONFIGS)}: {judge_name} for row {index+1} ---")
                    future = executor.submit(get_llm_judgment, row, judge_config)
                    future_to_judge_idx[future] = judge_idx

                for future in future_to_judge_idx:
                    judge_idx = future_to_judge_idx[future]
                    judge_config = JUDGE_CONFIGS[judge_idx]
                    judge_name = judge_config.get("name", f"Judge_{judge_idx}")
                    try:
                        score, reasoning = future.result() # Blocks until this judge completes
                        row_scores[judge_idx] = score
                        row_reasonings[judge_idx] = reasoning
                        individual_judge_results[f"{judge_name}_Score"].append(score)
                        individual_judge_results[f"{judge_name}_Reasoning"].append(reasoning)
                        logging.info(f"Judge {judge_name} for row {index+1} finished. Score: {score}")
                    except Exception as exc:
                        logging.error(f"Judge {judge_name} for row {index+1} generated an exception: {exc}")
                        # row_scores[judge_idx] and row_reasonings[judge_idx] remain None
                        individual_judge_results[f"{judge_name}_Score"].append(None)
                        individual_judge_results[f"{judge_name}_Reasoning"].append(f"Error during judgment: {exc}")


            # --- End of judges processing for the row ---

            # Calculate mean score for the row, ignoring None values
            valid_scores = [s for s in row_scores if s is not None]
            mean_score = np.mean(valid_scores) if valid_scores else None

            # Combine reasonings (simple join)
            combined_reasoning = "\n---\n".join(str(r) for r in row_reasonings if r is not None)

        aggregated_scores.append(mean_score)

        end_row_time = time.time()
        logging.info(f"Row {index+1} judged by all judges in {end_row_time - start_row_time:.2f} seconds. Mean Score: {mean_score if mean_score is not None else 'N/A'}")

        # Optional: Wait between rows (if judges have shared rate limits or to reduce load)
        if wait_time > 0 and index < num_rows_to_process - 1:
            logging.info(f"Waiting for {wait_time} seconds before next row...")
            time.sleep(wait_time)

    # --- Add results and Save ---
    logging.info("LLM judgment loop finished.")
    df_eval['Mean LLM Judge Score'] = aggregated_scores

    # Add individual judge results to the DataFrame
    for col_name, results_list in individual_judge_results.items():
         # Ensure list length matches DataFrame length (important if max_rows was used)
         if len(results_list) == len(df_eval):
              df_eval[col_name] = results_list
         else:
              logging.error(f"Length mismatch for column {col_name}. Expected {len(df_eval)}, got {len(results_list)}. Skipping column.")


    logging.info(f"Saving multi-judged evaluation results to {output_file}...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_eval.to_csv(output_file, index=False, encoding='utf-8')
        logging.info("Multi-judged evaluation results saved successfully.")

        # Display summary
        print("\n--- LLM Multi-Judgment Summary ---")
        print("Mean Score Distribution:")
        print(df_eval['Mean LLM Judge Score'].value_counts(dropna=False).sort_index())
        overall_avg_score = df_eval['Mean LLM Judge Score'].mean()
        print(f"\nOverall Average of Mean Scores: {overall_avg_score:.2f}" if not pd.isna(overall_avg_score) else "Overall Average of Mean Scores: N/A")

        # Print average score per judge
        print("\nAverage Score Per Judge:")
        for cfg in JUDGE_CONFIGS:
             judge_score_col = f"{cfg['name']}_Score"
             if judge_score_col in df_eval:
                  avg_judge_score = df_eval[judge_score_col].mean()
                  print(f"  - {cfg['name']}: {avg_judge_score:.2f}" if not pd.isna(avg_judge_score) else f"  - {cfg['name']}: N/A")


    except Exception as e:
        logging.error(f"Failed to save judged evaluation results: {e}", exc_info=True)

    judge_end_time = time.time()
    logging.info(f"Total LLM multi-judgment script execution time: {judge_end_time - judge_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG conclusions using multiple LLM judges.")
    parser.add_argument(
        "--input_file",
        default=EVALUATION_RESULTS_CSV_PATH,
        help=f"Path to the input evaluation results CSV (default: {EVALUATION_RESULTS_CSV_PATH})."
    )
    parser.add_argument(
        "--output_file",
        default=JUDGED_EVALUATION_RESULTS_CSV_PATH, # Use updated default
        help=f"Path to save the output CSV with multi-judge results (default: {JUDGED_EVALUATION_RESULTS_CSV_PATH})."
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None, # Process all rows by default
        help="Maximum number of rows to process from the input CSV (for debugging)."
    )
    parser.add_argument(
        "--wait_time",
        type=int,
        default=2, # Reduced default wait time between *judge* calls
        help="Seconds to wait between LLM API calls for each judge (adjust for rate limits)."
    )


    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        max_rows=args.max_rows,
        wait_time=args.wait_time
    )

# Ensure to host vLLM server before running this script
# Example command to start vLLM server:
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ~/visual-thinker-workspace/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31/ --tensor-parallel-size 4 --port 8000
# export OPENROUTER_API_KEY='your_actual_api_key_here' (no need for now)
# export GEMINI_API_KEY='your_actual_api_key_here'
# export DEEPINFRA_API_KEY='your_actual_api_key_here'
# export OPENAI_API_KEY="your_actual_api_key_here"

# Example usage from command line:
# python scripts/04_llm_judge_evaluation.py --input_file ../output/hybrid.csv --output_file ../output/judged_results.csv --max_rows 20 --wait_time 30