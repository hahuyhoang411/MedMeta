"""
Refactored Analysis Pipeline for Model Evaluation

This script consolidates multiple analysis tasks into a single, modular, and
configurable file. It allows for data preprocessing, individual model comparison
analysis, and detailed human-vs-LLM evaluation alignment studies.

Core Features:
- Centralized configuration for easy management of files and model names.
- Modular functions for each step of the analysis pipeline.
- Command-Line Interface (CLI) to run specific parts of the analysis or the
  entire pipeline.
- Elimination of redundant code by using generalized helper functions.

Usage (from the command line):
  
  # To see all available options
  python human_results_analyzer.py --help
  
  # To run only the initial data preprocessing step
  python human_results_analyzer.py --preprocess
  
  # To run all analyses for a specific comparison (e.g., Gemma Target vs. Gemma Direct)
  python human_results_analyzer.py --gemma-target-vs-direct
  
  # To run all basic comparison analyses (stats & t-tests)
  python human_results_analyzer.py --all-basic
  
  # To run all human-vs-LLM comparison analyses (correlations, ICC, plots)
  python human_results_analyzer.py --all-human-llm
  
  # To run the entire pipeline from start to finish
  python human_results_analyzer.py --all
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import json
import argparse
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Check for optional dependency 'pingouin' for advanced ICC calculation
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("Warning: 'pingouin' is not installed. Using a manual ICC calculation.")
    print("         For more detailed ICC statistics, run: pip install pingouin")


# =============================================================================
# 2. CONFIGURATION
# =============================================================================
# Centralized configuration for all file paths, model names, and parameters.
# This makes the script easy to update and maintain.
CONFIG = {
    'preprocess': {
        'evaluations_file': 'filtered_evaluations_noriz.csv',
        'tasks_file': 'tasks.json',
        'llm_eval_file': 'MedMeta_Combined_Conclusions.csv',
    },
    'medgemma_target_vs_direct': {
        'key': 'medgemma_target_vs_direct',
        'human_eval_file': 'medgemma_target_vs_medgemma_direct.csv',
        'model_a': 'MedGemma Target Conclusion',
        'model_b': 'MedGemma Direct Conclusion',
        'model_a_col': 'medgemma_target',
        'model_b_col': 'medgemma_direct',
        'llm_model_a_col': 'MedGemma_Target_Score',
        'llm_model_b_col': 'MedGemma_Direct_Score',
    },
    'gemma_target_vs_direct': {
        'key': 'gemma_target_vs_direct',
        'human_eval_file': 'gemma_target_vs_gemma_direct.csv',
        'model_a': 'Gemma Target Conclusion',
        'model_b': 'Gemma Direct Conclusion',
        'model_a_col': 'gemma_target',
        'model_b_col': 'gemma_direct',
        'llm_model_a_col': 'Gemma_Target_Score',
        'llm_model_b_col': 'Gemma_Direct_Score',
    },
    'medgemma_vs_gemma_target': {
        'key': 'medgemma_vs_gemma_target',
        'human_eval_file': 'medgemma_target_vs_gemma_target.csv',
        'model_a': 'MedGemma Target Conclusion',
        'model_b': 'Gemma Target Conclusion',
        'model_a_col': 'medgemma_target',
        'model_b_col': 'gemma_target',
        'llm_model_a_col': 'MedGemma_Target_Score',
        'llm_model_b_col': 'Gemma_Target_Score',
    },
    'medgemma_vs_gemma_direct': {
        'key': 'medgemma_vs_gemma_direct',
        'human_eval_file': 'medgemma_direct_vs_gemma_direct.csv',
        'model_a': 'MedGemma Direct Conclusion',
        'model_b': 'Gemma Direct Conclusion',
        'model_a_col': 'medgemma_direct',
        'model_b_col': 'gemma_direct',
        'llm_model_a_col': 'MedGemma_Direct_Score',
        'llm_model_b_col': 'Gemma_Direct_Score',
    },
}

# =============================================================================
# 3. HELPER & UTILITY FUNCTIONS (Refactored from original code)
# =============================================================================

def _load_dataframe(filepath):
    """Safely loads a CSV file into a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure the file exists, or run the preprocessing step first.")
        return None
    return pd.read_csv(filepath)

def _save_dataframe(df, filepath):
    """Saves a DataFrame to a CSV file and prints a confirmation."""
    df.to_csv(filepath, index=False)
    print(f"Successfully saved data to '{filepath}'")

def calculate_icc_manual(data1, data2):
    """
    Calculate ICC(2,1) - Two-way random effects, absolute agreement, single measurement.
    This function is preserved from your original script.
    """
    n = len(data1)
    if n < 2: return np.nan
    grand_mean = (np.mean(data1) + np.mean(data2)) / 2
    subject_means = (data1 + data2) / 2
    rater_means = [np.mean(data1), np.mean(data2)]
    all_scores = np.concatenate([data1, data2])
    SST = np.sum((all_scores - grand_mean) ** 2)
    SSB = 2 * np.sum((subject_means - grand_mean) ** 2)
    SSR = n * np.sum((rater_means - grand_mean) ** 2)
    SSE = SST - SSB - SSR
    MSB = SSB / (n - 1)
    MSR = SSR / 1  # 2 raters
    MSE = SSE / (n - 1)
    if (MSB + MSE + 2 * (MSR - MSE) / n) == 0: return np.nan
    icc = (MSB - MSE) / (MSB + MSE + 2 * (MSR - MSE) / n)
    return icc

def interpret_icc(icc_value):
    """Provides a qualitative interpretation of an ICC value."""
    if pd.isna(icc_value): return "Not applicable"
    if icc_value < 0.5: return "Poor reliability"
    if icc_value < 0.75: return "Moderate reliability"
    if icc_value < 0.9: return "Good reliability"
    return "Excellent reliability"

def get_comparison_type(row):
    """
    Create a comparison column to identify which comparison type each row belongs to.
    This function is preserved from your original script.
    """
    models = {row['modelA'], row['modelB']}
    if models == {'MedGemma Target Conclusion', 'MedGemma Direct Conclusion'}:
        return 'MedGemma Target vs MedGemma Direct'
    elif models == {'Gemma Target Conclusion', 'Gemma Direct Conclusion'}:
        return 'Gemma Target vs Gemma Direct'
    elif models == {'MedGemma Target Conclusion', 'Gemma Target Conclusion'}:
        return 'MedGemma Target vs Gemma Target'
    elif models == {'MedGemma Direct Conclusion', 'Gemma Direct Conclusion'}:
        return 'MedGemma Direct vs Gemma Direct'
    else:
        return 'Other'

def create_cross_task_comparison(df):
    """
    Create MedGemma Direct vs Gemma Direct comparison by combining results
    from different tasks with the same sourcePaperId.
    This function is preserved from your original script, but now takes the
    main DataFrame as an argument for better encapsulation.
    """
    medgemma_direct_evals = []
    gemma_direct_evals = []

    base_cols = ['annotator_id', 'sourcePaperId', 'sourcePaperTitle', 'task_id',
                 'session_start_time', 'evaluation_end_time', 'created_at', 'id']
    
    for _, row in df.iterrows():
        base_data = {col: row[col] for col in base_cols}
        if row['modelA'] == 'MedGemma Direct Conclusion':
            medgemma_direct_evals.append({**base_data, 'is_model_a': True, 'score': row['score_a']})
        if row['modelB'] == 'MedGemma Direct Conclusion':
            medgemma_direct_evals.append({**base_data, 'is_model_a': False, 'score': row['score_b']})
        if row['modelA'] == 'Gemma Direct Conclusion':
            gemma_direct_evals.append({**base_data, 'is_model_a': True, 'score': row['score_a']})
        if row['modelB'] == 'Gemma Direct Conclusion':
            gemma_direct_evals.append({**base_data, 'is_model_a': False, 'score': row['score_b']})

    medgemma_df = pd.DataFrame(medgemma_direct_evals)
    gemma_df = pd.DataFrame(gemma_direct_evals)
    
    cross_task_comparisons = []
    if medgemma_df.empty or gemma_df.empty:
        return pd.DataFrame()

    for source_paper_id in medgemma_df['sourcePaperId'].unique():
        if source_paper_id in gemma_df['sourcePaperId'].values:
            paper_medgemma = medgemma_df[medgemma_df['sourcePaperId'] == source_paper_id]
            paper_gemma = gemma_df[gemma_df['sourcePaperId'] == source_paper_id]
            
            num_pairs = min(3, len(paper_medgemma), len(paper_gemma))
            
            for i in range(num_pairs):
                medgemma_eval = paper_medgemma.iloc[i % len(paper_medgemma)]
                gemma_eval = paper_gemma.iloc[i % len(paper_gemma)]
                
                if medgemma_eval['is_model_a']:
                    model_a, model_b = 'MedGemma Direct Conclusion', 'Gemma Direct Conclusion'
                    score_a, score_b = medgemma_eval['score'], gemma_eval['score']
                    primary_eval = medgemma_eval
                else:
                    model_a, model_b = 'Gemma Direct Conclusion', 'MedGemma Direct Conclusion'
                    score_a, score_b = gemma_eval['score'], medgemma_eval['score']
                    primary_eval = gemma_eval

                comparison_row = {
                    'id': f"cross_{source_paper_id}_{i+1}",
                    'created_at': primary_eval['created_at'],
                    'annotator_id': f"cross_{medgemma_eval['annotator_id']}_{gemma_eval['annotator_id']}",
                    'task_id': f"cross_{source_paper_id}_{i+1}",
                    'score_a': score_a, 'score_b': score_b,
                    'session_start_time': primary_eval['session_start_time'],
                    'evaluation_end_time': max(medgemma_eval['evaluation_end_time'], gemma_eval['evaluation_end_time']),
                    'modelA': model_a, 'modelB': model_b,
                    'sourcePaperId': source_paper_id, 'sourcePaperTitle': primary_eval['sourcePaperTitle'],
                    'comparison_type': 'MedGemma Direct vs Gemma Direct'
                }
                cross_task_comparisons.append(comparison_row)
    
    return pd.DataFrame(cross_task_comparisons)

def _calculate_model_scores_generic(df, model_a_name, model_b_name, model_a_col, model_b_col):
    """
    A generalized version of the `calculate_model_scores` function.
    It takes model names and desired output column names as parameters.
    """
    results = []
    score_a_col_name = f"{model_a_col}_score"
    score_b_col_name = f"{model_b_col}_score"
    
    for _, row in df.iterrows():
        if row['modelA'] == model_a_name:
            model_a_score = row['score_a']
            model_b_score = row['score_b']
        else:
            model_a_score = row['score_b']
            model_b_score = row['score_a']
        
        base_data = row.to_dict()
        base_data[score_a_col_name] = model_a_score
        base_data[score_b_col_name] = model_b_score
        results.append(base_data)
        
    return pd.DataFrame(results)
    
def _calculate_human_model_scores_generic(df, model_a_name, model_b_name, model_a_col, model_b_col):
    """
    A generalized version of the `calculate_human_model_scores` function.
    """
    results = []
    human_score_a_col = f"human_{model_a_col}_score"
    human_score_b_col = f"human_{model_b_col}_score"
    
    for _, row in df.iterrows():
        if row['modelA'] == model_a_name:
            score_a = row['score_a']
            score_b = row['score_b']
        else:
            score_a = row['score_b']
            score_b = row['score_a']
            
        if score_a > score_b: preference = model_a_name
        elif score_b > score_a: preference = model_b_name
        else: preference = 'Tie'
            
        results.append({
            'PMID': row['sourcePaperId'],
            'task_id': row['task_id'],
            'annotator_id': row['annotator_id'],
            human_score_a_col: score_a,
            human_score_b_col: score_b,
            'human_preference': preference
        })
    return pd.DataFrame(results)


# =============================================================================
# 4. CORE ANALYSIS MODULES
# =============================================================================

def run_data_preprocessing():
    """
    Module 1: Load raw data, process it, and split it into separate CSV files
    for each comparison type. This corresponds to your first script block.
    """
    print("\n" + "="*80)
    print("RUNNING MODULE 1: DATA PREPROCESSING AND SPLITTING")
    print("="*80)

    # --- Load Data ---
    conf = CONFIG['preprocess']
    df = _load_dataframe(conf['evaluations_file'])
    if df is None: return

    try:
        with open(conf['tasks_file'], 'r') as f:
            tasks_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{conf['tasks_file']}' was not found.")
        return
        
    print(f"Loaded {len(df)} rows from {conf['evaluations_file']}")
    
    # --- Process and Map Task Data ---
    task_mapping = {
        task['taskId']: {
            'modelA': task['modelIdentities']['modelA'],
            'modelB': task['modelIdentities']['modelB'],
            'sourcePaperId': task['sourcePaperId'],
            'sourcePaperTitle': task['sourcePaperTitle']
        } for task in tasks_data.get('evaluationTasks', [])
    }
    
    for key in ['modelA', 'modelB', 'sourcePaperId', 'sourcePaperTitle']:
        df[key] = df['task_id'].map(lambda x: task_mapping.get(x, {}).get(key))

    # --- Identify and Split by Comparison Type ---
    df['comparison_type'] = df.apply(get_comparison_type, axis=1)
    
    comparison_dfs = {}
    comparison_dfs['medgemma_target_vs_direct'] = df[df['comparison_type'] == 'MedGemma Target vs MedGemma Direct'].copy()
    comparison_dfs['gemma_target_vs_direct'] = df[df['comparison_type'] == 'Gemma Target vs Gemma Direct'].copy()
    comparison_dfs['medgemma_vs_gemma_target'] = df[df['comparison_type'] == 'MedGemma Target vs Gemma Target'].copy()
    
    # --- Handle Cross-Task Comparison ---
    medgemma_direct_vs_gemma_direct = df[df['comparison_type'] == 'MedGemma Direct vs Gemma Direct'].copy()
    cross_task_df = create_cross_task_comparison(df)
    
    all_medgemma_direct_vs_gemma_direct = pd.concat(
        [medgemma_direct_vs_gemma_direct, cross_task_df], ignore_index=True
    )
    comparison_dfs['medgemma_vs_gemma_direct'] = all_medgemma_direct_vs_gemma_direct

    # --- Save Results and Print Summary ---
    print("\n--- Summary of Split Data ---")
    for key, c_df in comparison_dfs.items():
        output_file = CONFIG[key]['human_eval_file']
        _save_dataframe(c_df, output_file)
        print(f"  - {CONFIG[key]['model_a']} vs {CONFIG[key]['model_b']}: {len(c_df)} rows")
    
    print(f"Cross-task comparisons created: {len(cross_task_df)} rows")
    print("\nPreprocessing complete.")


def run_basic_comparison_analysis(conf):
    """
    Module 2: Perform a basic statistical analysis on a comparison file.
    This includes overall scores, per-paper scores, and a paired t-test.
    """
    print("\n" + "="*80)
    print(f"RUNNING MODULE 2: BASIC ANALYSIS for {conf['key'].upper()}")
    print(f"Models: {conf['model_a']} vs. {conf['model_b']}")
    print("="*80)
    
    df = _load_dataframe(conf['human_eval_file'])
    if df is None or df.empty:
        print(f"Skipping analysis for {conf['key']} due to missing or empty data.")
        return

    # --- Calculate Model-Specific Scores ---
    model_scores_df = _calculate_model_scores_generic(
        df, conf['model_a'], conf['model_b'], conf['model_a_col'], conf['model_b_col']
    )
    
    score_a_col = f"{conf['model_a_col']}_score"
    score_b_col = f"{conf['model_b_col']}_score"
    
    # --- Overall Average Scores ---
    overall_avg_a = model_scores_df[score_a_col].mean()
    overall_avg_b = model_scores_df[score_b_col].mean()

    print("\n=== OVERALL AVERAGE SCORES ===")
    print(f"{conf['model_a']}: {overall_avg_a:.3f}")
    print(f"{conf['model_b']}: {overall_avg_b:.3f}")
    print(f"Total evaluations: {len(model_scores_df)}")

    # --- Average Scores by Source Paper ID ---
    paper_scores = model_scores_df.groupby('sourcePaperId').agg({
        score_a_col: ['mean', 'count'],
        score_b_col: 'mean',
    }).round(3)
    paper_scores.columns = [f"{conf['model_a_col']}_avg", f"{conf['model_a_col']}_count", f"{conf['model_b_col']}_avg"]
    paper_scores['score_difference'] = paper_scores[f"{conf['model_a_col']}_avg"] - paper_scores[f"{conf['model_b_col']}_avg"]

    print("\n=== AVERAGE SCORES BY SOURCE PAPER ID ===")
    print(paper_scores.to_string())

    # --- Summary Statistics ---
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Number of unique papers: {len(paper_scores)}")
    print(f"Papers where {conf['model_a']} scored higher: {(paper_scores['score_difference'] > 0).sum()}")
    print(f"Papers where {conf['model_b']} scored higher: {(paper_scores['score_difference'] < 0).sum()}")
    print(f"Papers with tied scores: {(paper_scores['score_difference'] == 0).sum()}")

    # --- Statistical Significance ---
    if len(paper_scores) > 1:
        t_stat, p_value = stats.ttest_rel(paper_scores[f"{conf['model_a_col']}_avg"], paper_scores[f"{conf['model_b_col']}_avg"])
        print("\n=== STATISTICAL ANALYSIS (PAIRED T-TEST) ===")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.3f}")
        print(f"Significant difference (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")
    
    print(f"\nBasic analysis for {conf['key']} complete.")


def run_human_vs_llm_analysis(conf):
    """
    Module 3: Perform a detailed comparison between human and LLM evaluations.
    This includes correlation, agreement, ICC, and visualizations.
    """
    print("\n" + "="*80)
    print(f"RUNNING MODULE 3: HUMAN vs. LLM ANALYSIS for {conf['key'].upper()}")
    print(f"Models: {conf['model_a']} vs. {conf['model_b']}")
    print("="*80)
    
    # --- Load Data ---
    human_eval = _load_dataframe(conf['human_eval_file'])
    llm_eval_file = CONFIG['preprocess']['llm_eval_file']
    llm_eval = _load_dataframe(llm_eval_file)
    
    if human_eval is None or llm_eval is None or human_eval.empty or llm_eval.empty:
        print(f"Skipping analysis for {conf['key']} due to missing or empty data.")
        return

    # --- Process Human Scores ---
    human_scores = _calculate_human_model_scores_generic(
        human_eval, conf['model_a'], conf['model_b'], conf['model_a_col'], conf['model_b_col']
    )
    human_avg_by_pmid = human_scores.groupby('PMID').agg({
        f"human_{conf['model_a_col']}_score": 'mean',
        f"human_{conf['model_b_col']}_score": 'mean',
    }).round(3)
    human_preferences = human_scores.groupby('PMID')['human_preference'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'Tie'
    ).reset_index()
    human_summary = human_avg_by_pmid.merge(human_preferences, on='PMID')
    
    # --- Process LLM Scores ---
    llm_processed = llm_eval[['PMID', conf['llm_model_a_col'], conf['llm_model_b_col']]].copy()
    llm_processed.columns = ['PMID', f"llm_{conf['model_a_col']}_score", f"llm_{conf['model_b_col']}_score"]
    
    llm_processed['llm_preference'] = np.select(
        [
            llm_processed[f"llm_{conf['model_a_col']}_score"] > llm_processed[f"llm_{conf['model_b_col']}_score"],
            llm_processed[f"llm_{conf['model_b_col']}_score"] > llm_processed[f"llm_{conf['model_a_col']}_score"]
        ],
        [conf['model_a'], conf['model_b']],
        default='Tie'
    )
    
    # --- Merge Data ---
    comparison_data = human_summary.merge(llm_processed, on='PMID', how='inner')
    if comparison_data.empty:
        print("No matching PMIDs found between human and LLM evaluations. Cannot proceed.")
        return
        
    print(f"Found {len(comparison_data)} matching PMIDs for comparison.")

    # --- Define Score Columns for Convenience ---
    h_score_a = f"human_{conf['model_a_col']}_score"
    h_score_b = f"human_{conf['model_b_col']}_score"
    l_score_a = f"llm_{conf['model_a_col']}_score"
    l_score_b = f"llm_{conf['model_b_col']}_score"

    # --- Analysis ---
    print("\n--- CORRELATION ANALYSIS (PEARSON) ---")
    corr_a = stats.pearsonr(comparison_data[h_score_a], comparison_data[l_score_a])
    corr_b = stats.pearsonr(comparison_data[h_score_b], comparison_data[l_score_b])
    print(f"{conf['model_a']} Score Correlation: r={corr_a[0]:.3f}, p={corr_a[1]:.3f}")
    print(f"{conf['model_b']} Score Correlation: r={corr_b[0]:.3f}, p={corr_b[1]:.3f}")

    print("\n--- PREFERENCE AGREEMENT ANALYSIS ---")
    agreement = (comparison_data['human_preference'] == comparison_data['llm_preference']).mean()
    print(f"Overall Agreement: {agreement:.2%}")
    print("Agreement Table (Human vs. LLM):")
    print(pd.crosstab(comparison_data['human_preference'], comparison_data['llm_preference']))

    print("\n--- ICC ANALYSIS (RELIABILITY) ---")
    icc_a, icc_b = np.nan, np.nan
    if PINGOUIN_AVAILABLE:
        try:
            icc_data_a = pd.DataFrame({'Subject': comparison_data['PMID'], 'Human': comparison_data[h_score_a], 'LLM': comparison_data[l_score_a]})
            icc_a = pg.intraclass_corr(data=icc_data_a, targets='Subject', raters=['Human', 'LLM'], ratings='value').set_index('Type').loc['ICC2', 'ICC']
            icc_data_b = pd.DataFrame({'Subject': comparison_data['PMID'], 'Human': comparison_data[h_score_b], 'LLM': comparison_data[l_score_b]})
            icc_b = pg.intraclass_corr(data=icc_data_b, targets='Subject', raters=['Human', 'LLM'], ratings='value').set_index('Type').loc['ICC2', 'ICC']
        except Exception: # Fallback on error
            icc_a = calculate_icc_manual(comparison_data[h_score_a], comparison_data[l_score_a])
            icc_b = calculate_icc_manual(comparison_data[h_score_b], comparison_data[l_score_b])
    else:
        icc_a = calculate_icc_manual(comparison_data[h_score_a], comparison_data[l_score_a])
        icc_b = calculate_icc_manual(comparison_data[h_score_b], comparison_data[l_score_b])
    
    print(f"{conf['model_a']} ICC(2,1): {icc_a:.3f} ({interpret_icc(icc_a)})")
    print(f"{conf['model_b']} ICC(2,1): {icc_b:.3f} ({interpret_icc(icc_b)})")

    # --- Visualization ---
    print("\n--- VISUALIZATION ---")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Human vs. LLM Evaluation: {conf['model_a']} vs. {conf['model_b']}", fontsize=16)

    axes[0, 0].scatter(comparison_data[h_score_a], comparison_data[l_score_a], alpha=0.6)
    axes[0, 0].set_title(f"{conf['model_a']} Score Correlation (r={corr_a[0]:.3f})")
    axes[0, 0].set_xlabel('Human Score'); axes[0, 0].set_ylabel('LLM Score')
    axes[0, 0].plot([1, 5], [1, 5], 'r--', alpha=0.5)

    axes[0, 1].scatter(comparison_data[h_score_b], comparison_data[l_score_b], alpha=0.6)
    axes[0, 1].set_title(f"{conf['model_b']} Score Correlation (r={corr_b[0]:.3f})")
    axes[0, 1].set_xlabel('Human Score'); axes[0, 1].set_ylabel('LLM Score')
    axes[0, 1].plot([1, 5], [1, 5], 'r--', alpha=0.5)

    comparison_data['human_score_diff'] = comparison_data[h_score_a] - comparison_data[h_score_b]
    comparison_data['llm_score_diff'] = comparison_data[l_score_a] - comparison_data[l_score_b]
    diff_corr = stats.pearsonr(comparison_data['human_score_diff'], comparison_data['llm_score_diff'])
    axes[1, 0].scatter(comparison_data['human_score_diff'], comparison_data['llm_score_diff'], alpha=0.6)
    axes[1, 0].set_title(f"Score Difference Correlation (r={diff_corr[0]:.3f})")
    axes[1, 0].set_xlabel(f'Human Diff ({conf["model_a_col"]} - {conf["model_b_col"]})')
    axes[1, 0].set_ylabel(f'LLM Diff ({conf["model_a_col"]} - {conf["model_b_col"]})')
    axes[1, 0].axhline(0, color='k', ls='--', alpha=0.4); axes[1, 0].axvline(0, color='k', ls='--', alpha=0.4)

    sns.heatmap(pd.crosstab(comparison_data['human_preference'], comparison_data['llm_preference'], normalize='index'),
                annot=True, cmap='Blues', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title('Preference Agreement Heatmap')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot and data
    plot_filename = f"{conf['key']}_human_llm_comparison.png"
    data_filename = f"{conf['key']}_human_llm_comparison_data.csv"
    plt.savefig(plot_filename, dpi=300)
    _save_dataframe(comparison_data, data_filename)
    
    print(f"Plot saved to '{plot_filename}'")
    print(f"Detailed comparison data saved to '{data_filename}'")
    plt.show()

    print(f"\nHuman vs. LLM analysis for {conf['key']} complete.")


# =============================================================================
# 5. MAIN EXECUTION BLOCK (CLI)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A modular pipeline for analyzing model evaluation data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--preprocess', 
        action='store_true',
        help='Run Module 1: Preprocess raw data and split into comparison CSVs.'
    )
    
    # Dynamically create arguments for each comparison type
    for key, conf in CONFIG.items():
        if key == 'preprocess': continue
        parser.add_argument(
            f'--{key}',
            action='store_true',
            help=f"Run all analyses for '{conf['model_a']}' vs. '{conf['model_b']}'."
        )

    parser.add_argument(
        '--all-basic', 
        action='store_true',
        help='Run Module 2 (Basic Analysis) for all comparison types.'
    )
    parser.add_argument(
        '--all-human-llm', 
        action='store_true',
        help='Run Module 3 (Human vs. LLM Analysis) for all comparison types.'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run the entire pipeline: preprocess, then all analyses.'
    )

    args = parser.parse_args()
    
    # Check if any argument was passed
    if not any(vars(args).values()):
        parser.print_help()
        print("\nNo analysis selected. Please specify a module to run (e.g., --preprocess or --all).")
    else:
        if args.all or args.preprocess:
            run_data_preprocessing()
            
        analysis_keys = [k for k in CONFIG if k != 'preprocess']

        for key in analysis_keys:
            run_basic = args.all or args.all_basic or getattr(args, key.replace('-', '_'), False)
            run_human_llm = args.all or args.all_human_llm or getattr(args, key.replace('-', '_'), False)

            if run_basic:
                run_basic_comparison_analysis(CONFIG[key])
            if run_human_llm:
                run_human_vs_llm_analysis(CONFIG[key])
        
        print("\n" + "="*80)
        print("All requested tasks are complete.")
        print("="*80)
