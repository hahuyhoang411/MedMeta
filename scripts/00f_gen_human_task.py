import pandas as pd
import json
import random
from pathlib import Path
import ast
import numpy as np
from typing import List, Dict, Any, Tuple

# ... (load_and_preprocess_data remains the same as the previous version) ...

def load_and_preprocess_data(
    conclusions_path: Path, abstracts_path: Path
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Loads and preprocesses all necessary data files."""
    print("Phase 1: Loading and preprocessing data...")
    conclusions_df = pd.read_csv(conclusions_path)
    conclusions_df['Target PMIDs (for eval metrics)'] = conclusions_df['Target PMIDs (for eval metrics)'].apply(ast.literal_eval)
    abstracts_df = pd.read_csv(abstracts_path)
    abstract_lookup = pd.Series(abstracts_df.text.values, index=abstracts_df.PMID).to_dict()
    # Scores are now merged into the conclusions file, so no need to load a separate score file.
    full_df = conclusions_df
    print("Data loading and preprocessing complete.")
    return full_df, abstract_lookup


# ==============================================================================
#  NEW Phase 2: Hypothesis-Driven Stratified Sampling
# ==============================================================================

def perform_hypothesis_driven_sampling(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Selects a representative subset using hypothesis-driven stratification.

    Methodology:
    1.  Calculates four new metrics for each source, each corresponding to a
        key research hypothesis.
    2.  Normalizes these metrics into percentiles (0 to 1) to make them comparable.
    3.  Assigns each source to a primary stratum based on its highest normalized
        metric. This identifies the "most interesting feature" of each source.
    4.  Performs proportional sampling from these new, meaningful strata.
    """
    print("\nPhase 2: Performing Hypothesis-Driven Stratified Sampling...")

    # --- Step 1: Calculate Hypothesis-Driven Metrics ---
    # Metric for H1 (Domain-Adaptation)
    df['h1_metric'] = df['MedGemma_Target_Score'] - df['Gemma_Target_Score']
    
    # Metric for H2 (Information Grounding)
    h2_medgemma_gain = df['MedGemma_Target_Score'] - df['MedGemma_Direct_Score']
    h2_gemma_gain = df['Gemma_Target_Score'] - df['Gemma_Direct_Score']
    df['h2_metric'] = (h2_medgemma_gain + h2_gemma_gain) / 2
    
    # Metric for H3 (Interaction Effect)
    df['h3_metric'] = h2_gemma_gain - h2_medgemma_gain
    
    # Metric for H5 (Agreement/Ambiguity - lower is more interesting)
    score_cols = ['MedGemma_Target_Score', 'MedGemma_Direct_Score', 'Gemma_Target_Score', 'Gemma_Direct_Score']
    df['h5_metric'] = df[score_cols].std(axis=1)

    # --- Step 2: Normalize metrics to be comparable ---
    # We rank them and convert to percentiles. For std dev, lower is more interesting,
    # so we invert the rank.
    df['h1_norm'] = df['h1_metric'].rank(pct=True)
    df['h2_norm'] = df['h2_metric'].rank(pct=True)
    df['h3_norm'] = df['h3_metric'].rank(pct=True)
    df['h5_norm'] = df['h5_metric'].rank(pct=True, ascending=False) # Inverted

    # --- Step 3: Assign a primary stratum ---
    metric_cols = ['h1_norm', 'h2_norm', 'h3_norm', 'h5_norm']
    df['primary_stratum'] = df[metric_cols].idxmax(axis=1).str.replace('_norm', '')
    
    stratum_map = {
        'h1': 'Domain_Advantage',
        'h2': 'Grounding_Advantage',
        'h3': 'Interaction_Effect',
        'h5': 'High_Agreement'
    }
    df['primary_stratum'] = df['primary_stratum'].map(stratum_map)
    
    print("\nStratum distribution in full population:")
    print(df['primary_stratum'].value_counts(normalize=True))

    # --- Step 4: Proportional Sampling ---
    sampled_df = df.groupby('primary_stratum', group_keys=False).apply(
        lambda x: x.sample(int(np.rint(n_samples * len(x) / len(df))))
    )
    
    # Adjust if rounding caused a mismatch
    missing_count = n_samples - len(sampled_df)
    if missing_count > 0:
        pool = df.drop(sampled_df.index)
        sampled_df = pd.concat([sampled_df, pool.sample(missing_count, random_state=1)]) # Use random_state for reproducibility
    
    print(f"\nSelected {len(sampled_df)} sources for the evaluation set via hypothesis-driven strata.")
    return sampled_df.drop(columns=[col for col in df.columns if '_metric' in col or '_norm' in col or 'primary_stratum' in col])


# ... The rest of the script (create_task_pool, create_final_balanced_list, main)
# remains exactly the same as the previous version. You just need to swap out
# the `perform_stratified_sampling` function call in `main` with this new one.

def create_task_pool(
    sampled_df: pd.DataFrame, abstract_lookup: Dict[int, str]
) -> List[Dict[str, Any]]:
    # This function does not need to change
    # ... (code from previous version) ...
    print("\nPhase 3: Generating unique task pool from the sampled sources...")
    unique_tasks = []
    
    comparison_pairs = [
        ("MedGemma Target Conclusion", "MedGemma Direct Conclusion"),
        ("Gemma Target Conclusion", "Gemma Direct Conclusion"),
        ("MedGemma Target Conclusion", "Gemma Target Conclusion"),
    ]

    for _, row in sampled_df.iterrows():
        pmids = row['Target PMIDs (for eval metrics)']
        source_abstracts = [
            abstract_lookup.get(pmid, f"Error: Abstract for PMID {pmid} not found.") for pmid in pmids
        ]

        for pair in comparison_pairs:
            model1_name, model2_name = pair
            
            models = [
                {"name": model1_name, "text": row[model1_name]},
                {"name": model2_name, "text": row[model2_name]}
            ]
            random.shuffle(models)
            
            model_a, model_b = models[0], models[1]

            task = {
                "sourcePaperId": int(row['PMID']),
                "metaAnalysisName": row['Meta Analysis Name'],
                "referenceConclusion": row['Original Conclusion'],
                "sourceAbstracts": source_abstracts,
                "modelOutputs": {
                    "conclusionA": model_a["text"], "conclusionB": model_b["text"],
                },
                "modelIdentities": {
                    "modelA": model_a["name"], "modelB": model_b["name"],
                }
            }
            unique_tasks.append(task)
    
    print(f"Generated {len(unique_tasks)} unique tasks.")
    return unique_tasks


def create_final_balanced_list(
    unique_tasks: List[Dict[str, Any]], ratings_per_task: int
) -> List[Dict[str, Any]]:
    # This function does not need to change
    # ... (code from previous version) ...
    print(f"\nPhase 4: Creating final balanced list with {ratings_per_task} ratings per unique task...")
    
    final_task_list = []
    for _ in range(ratings_per_task):
        final_task_list.extend([task.copy() for task in unique_tasks])
        
    random.shuffle(final_task_list)
    
    for i, task in enumerate(final_task_list):
        task['taskId'] = i + 1
        
    print(f"Generated final list of {len(final_task_list)} total annotation tasks.")
    return final_task_list


def main():
    """Main function to orchestrate the task generation process."""
    
    # --- Configuration ---
    N_SAMPLES_TO_SELECT = 20
    RATINGS_PER_TASK = 3
    
    # --- File Paths ---
    CONCLUSIONS_FILE = Path("MedMeta_Combined_Conclusions.csv")
    ABSTRACTS_FILE = Path("pubmed_data_reference_eval_new_sampling.csv")
    OUTPUT_FILE = Path("tasks.json")
    
    # ... (file checking logic) ...

    # Execute the pipeline
    full_df, abstract_lookup = load_and_preprocess_data(CONCLUSIONS_FILE, ABSTRACTS_FILE)
    # The only change is calling the new sampling function
    sampled_df = perform_hypothesis_driven_sampling(full_df, N_SAMPLES_TO_SELECT)
    unique_tasks = create_task_pool(sampled_df, abstract_lookup)
    final_task_list = create_final_balanced_list(unique_tasks, RATINGS_PER_TASK)
    
    print(f"\nPhase 5: Writing {len(final_task_list)} tasks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_task_list, f, indent=2, ensure_ascii=False)
        
    print("\nSUCCESS! `tasks.json` has been created with a robust, hypothesis-driven experimental design.")

if __name__ == "__main__":
    main()