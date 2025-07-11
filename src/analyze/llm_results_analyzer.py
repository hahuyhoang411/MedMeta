# llm_results_analyzer.py
"""
Master Analysis Suite

This script consolidates a series of data analysis and visualization modules for evaluating
Large Language Model (LLM) performance. It is designed to be run from the command line,
allowing users to selectively execute different parts of the analysis.

Core Features:
- Loads and processes evaluation data from CSV files.
- Provides a suite of analyses, including:
  - Overall performance summaries and statistical tests (t-tests).
  - Inter-judge agreement and similarity (Spearman, Cohen's Kappa).
  - Performance breakdown by various factors (year, topic, approach).
  - Semantic similarity scoring using BERTScore.
- Generates and saves plots and summary tables to output directories.

---------------------------------------------------------------------------------
Command-Line Usage:
---------------------------------------------------------------------------------

You must specify which analysis you want to run.

--- Basic Examples:
# Run only the BERTScore analysis
> python main_analyzer.py --bertscore

# Run the judge similarity and kappa agreement analyses
> python main_analyzer.py --similarity --kappa

# Run all standard analyses and save outputs to a specific directory
> python main_analyzer.py --all --output-dir ./analysis_results

--- Listing all options:
> python main_analyzer.py --help

---------------------------------------------------------------------------------
"""

# --- 1. IMPORTS ---
# Standard library imports
import argparse
import json
import os
import re
from itertools import combinations

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evaluate import load as hf_load
from scipy.stats import ttest_rel
from sklearn.metrics import (cohen_kappa_score, mean_absolute_error,
                             mean_squared_error)
from tqdm import tqdm

# --- 2. CONFIGURATION & CONSTANTS ---

# Mapping for cleaning up judge column names in the input CSVs
JUDGE_COL_MAP = {
    'OpenAI-o4-mini-high_Score': 'O4-mini_Score',
    'Qwen3-235B-A22B_Score': 'Qwen3-235B_Score',
    'Gemini-Pro-Low_Score': 'Gemini-Pro-2.5_Score'
}

# Configuration for BERTScore analysis
BERTSCORE_MODEL_TYPE = "answerdotai/ModernBERT-base"
BERTSCORE_NUM_LAYERS = 18  # Specific to ModernBERT with 22 layers
BERTSCORE_LANG = "en"
BERTSCORE_MODEL_INSTANCE = None # Global variable to hold the loaded model

# Constants for statistical tests
MIN_SAMPLES_FOR_TTEST = 5
MIN_SAMPLES_FOR_BERTSCORE = 5


# --- 3. CORE DATA UTILITIES ---
# These functions are used by multiple analysis modules to load and prepare data.

def get_csv_files(directory="."):
    """Lists all CSV files in the specified directory matching the '04_' pattern."""
    files = [f for f in os.listdir(directory) if f.startswith("04_") and f.endswith(".csv")]
    if not files:
        print(f"Warning: No CSV files starting with '04_' found in directory '{directory}'.")
    return files

def parse_filename(filename):
    """
    Parses model name and approach from the filename.
    Example: "04_flash_direct.csv" -> ("flash", "direct")
    """
    name_part = filename[3:-4]
    parts = name_part.split('_')
    known_approaches = ["direct", "target", "k10", "k5", "target_negated"]

    if not parts:
        return None, None

    # Sort approaches by length (desc) to match longer names first (e.g., 'target_negated' before 'target')
    known_approaches_sorted = sorted(known_approaches, key=len, reverse=True)

    for approach in known_approaches_sorted:
        if name_part.endswith('_' + approach):
            model_name = name_part[:-len('_' + approach)]
            return model_name if model_name else None, approach

    return name_part, None

def load_all_data(directory="."):
    """
    Loads all relevant CSV files into a single, raw, long-format Pandas DataFrame.
    Includes columns for score analysis (mean and individual judges) and text columns.
    """
    csv_files = get_csv_files(directory)
    all_data_list = []

    for f_name in csv_files:
        file_path = os.path.join(directory, f_name)
        model, approach = parse_filename(f_name)

        if model is None or approach is None:
            print(f"Skipping file due to parsing error: {f_name}")
            continue

        try:
            df_temp = pd.read_csv(file_path, index_col="Number")

            cols_to_select_and_rename = {
                'Mean LLM Judge Score': 'Score',
                'Cleaned Year': 'Year',
                'Topic': 'Topic',
                'Suitability Score': 'Suitability Score',
                'Conclusion': 'Reference_Conclusion',
                'Generated Conclusion': 'Predicted_Conclusion'
            }
            cols_to_select_and_rename.update(JUDGE_COL_MAP)

            current_data = {}
            for original_col, new_col_name in cols_to_select_and_rename.items():
                if original_col in df_temp.columns:
                    current_data[new_col_name] = df_temp[original_col]

            df_final_temp = pd.DataFrame(current_data, index=df_temp.index)

            for col in ['Reference_Conclusion', 'Predicted_Conclusion']:
                if col not in df_final_temp.columns:
                    df_final_temp[col] = pd.NA

            df_final_temp['Model'] = model
            df_final_temp['Approach'] = approach

            df_final_temp.reset_index(inplace=True)
            all_data_list.append(df_final_temp)

        except Exception as e:
            print(f"Error loading or processing file {f_name}: {e}")

    if not all_data_list:
        print("No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_list, ignore_index=True)

    # Type conversions
    combined_df['Year'] = pd.to_numeric(combined_df['Year'], errors='coerce')
    combined_df.dropna(subset=['Year'], inplace=True)
    combined_df['Year'] = combined_df['Year'].astype(int)
    combined_df['Suitability Score'] = pd.to_numeric(combined_df['Suitability Score'], errors='coerce')
    combined_df['Reference_Conclusion'] = combined_df['Reference_Conclusion'].astype(pd.StringDtype())
    combined_df['Predicted_Conclusion'] = combined_df['Predicted_Conclusion'].astype(pd.StringDtype())

    for new_col_name in JUDGE_COL_MAP.values():
        if new_col_name in combined_df.columns:
            combined_df[new_col_name] = pd.to_numeric(combined_df[new_col_name], errors='coerce')

    return combined_df

def process_data_for_analysis(df_all_raw):
    """
    Pivots the raw data to a wide format for improvement analysis.
    This creates columns like 'k5_improvement' (k5 score - direct score).
    """
    if df_all_raw.empty:
        return pd.DataFrame()

    all_models_data_processed = []

    for model_name in df_all_raw['Model'].unique():
        df_model_raw = df_all_raw[df_all_raw['Model'] == model_name].copy()
        pivot_idx_cols = ['Number', 'Year', 'Topic', 'Suitability Score']
        valid_pivot_idx_cols = [col for col in pivot_idx_cols if col in df_model_raw.columns]

        if not valid_pivot_idx_cols or 'Number' not in valid_pivot_idx_cols:
            continue

        try:
            df_pivot = df_model_raw.pivot_table(index=valid_pivot_idx_cols,
                                                columns='Approach',
                                                values='Score').reset_index()
        except Exception as e:
            print(f"Could not pivot data for model '{model_name}': {e}")
            continue

        if 'direct' in df_pivot.columns:
            for approach_suffix in ['k5', 'k10', 'target', 'target_negated']:
                if approach_suffix in df_pivot.columns:
                    df_pivot[f'{approach_suffix}_improvement'] = df_pivot[approach_suffix] - df_pivot['direct']
                else:
                    df_pivot[f'{approach_suffix}_improvement'] = pd.NA
        else:
            for approach_suffix in ['k5', 'k10', 'target', 'target_negated']:
                 df_pivot[f'{approach_suffix}_improvement'] = pd.NA

        df_pivot['Model'] = model_name
        all_models_data_processed.append(df_pivot)

    if not all_models_data_processed:
        return pd.DataFrame()

    return pd.concat(all_models_data_processed, ignore_index=True)


# --- 4. ANALYSIS & PLOTTING MODULES ---

# === Module: Judge Score Comparisons ===

def plot_model_comparison_overlapped(df_raw, output_dir):
    """
    For each approach, creates a bar chart where bars for each judge are
    overlapped with transparency, with trendlines drawn over them.
    """
    print("\n--- Running: Model Comparison Overlapped Plot ---")
    if df_raw.empty:
        print("Raw DataFrame is empty. Skipping plot.")
        return

    judge_cols = list(JUDGE_COL_MAP.values())
    available_judge_cols = [col for col in judge_cols if col in df_raw.columns]
    if not available_judge_cols:
        print(f"No individual judge score columns found. Looked for: {judge_cols}")
        return

    os.makedirs(output_dir, exist_ok=True)
    mean_scores = df_raw.groupby(['Model', 'Approach'])[available_judge_cols].mean().reset_index()
    approaches = mean_scores['Approach'].unique()

    for approach in approaches:
        data_for_plot = mean_scores[mean_scores['Approach'] == approach].copy().sort_values('Model')
        if data_for_plot.empty:
            continue

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 9))

        judges = [col.replace('_Score', '') for col in available_judge_cols]
        colors = sns.color_palette("muted", n_colors=len(judges))
        judge_color_map = dict(zip(judges, colors))

        for judge in judges:
            judge_score_col = f'{judge}_Score'
            sns.barplot(x='Model', y=judge_score_col, data=data_for_plot,
                        color=judge_color_map[judge], alpha=0.5, label=judge, ax=ax)
            sns.lineplot(x='Model', y=judge_score_col, data=data_for_plot,
                         color=judge_color_map[judge], marker='o', markersize=8,
                         linewidth=2.5, ax=ax)

        ax.set_title(f"Judge Score Comparison Across Models for Approach: '{approach.upper()}'", fontsize=20, pad=20)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Mean Score', fontsize=14)
        ax.tick_params(axis='x', labelsize=12, rotation=25)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(0)

        handles, labels = ax.get_legend_handles_labels()
        solid_handles = [plt.Rectangle((0, 0), 1, 1, color=judge_color_map[label]) for label in labels]
        ax.legend(solid_handles, labels, title='Judge', fontsize='12', loc='upper right')
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f"model_comparison_overlapped_{approach}.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot: {plot_filename}")
        plt.close(fig)

def analyze_and_plot_similarity(df_raw, output_dir):
    """
    Calculates and visualizes inter-judge correlation (Spearman) for each approach.
    """
    print("\n--- Running: Judge Similarity Analysis (Spearman Correlation) ---")
    if df_raw.empty:
        print("Raw DataFrame is empty. Skipping analysis.")
        return

    judge_cols = list(JUDGE_COL_MAP.values())
    available_judge_cols = [col for col in judge_cols if col in df_raw.columns]
    if len(available_judge_cols) < 2:
        print("Need at least two judge score columns to calculate correlation. Aborting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    mean_scores = df_raw.groupby(['Model', 'Approach'])[available_judge_cols].mean()
    approaches = mean_scores.index.get_level_values('Approach').unique()

    for approach in approaches:
        print(f"\n--- Analysis for Approach: '{approach.upper()}' ---")
        df_approach_scores = mean_scores.loc[(slice(None), approach), :]
        df_approach_scores.index = df_approach_scores.index.get_level_values('Model')
        df_approach_scores.columns = [col.replace('_Score', '') for col in df_approach_scores.columns]
        correlation_matrix = df_approach_scores.corr(method='spearman')

        print("Spearman Correlation Matrix:")
        print(correlation_matrix.to_string(float_format="%.3f"))

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='vlag', fmt='.2f', linewidths=.5, vmin=-1, vmax=1)
        plt.title(f"Inter-Judge Score Correlation\nApproach: '{approach.upper()}'", fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f"judge_correlation_{approach}.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved heatmap: {plot_filename}")
        plt.close()

def analyze_and_plot_kappa(df_raw, output_dir):
    """
    Calculates and visualizes inter-judge agreement using Weighted Cohen's Kappa.
    """
    print("\n--- Running: Judge Agreement Analysis (Cohen's Kappa) ---")
    if df_raw.empty:
        print("Raw DataFrame is empty. Skipping analysis.")
        return

    judge_cols = list(JUDGE_COL_MAP.values())
    available_judge_cols = [col for col in judge_cols if col in df_raw.columns]
    if len(available_judge_cols) < 2:
        print("Need at least two judge score columns for Kappa. Aborting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    approaches = df_raw['Approach'].unique()

    for approach in approaches:
        print(f"\n--- Kappa Analysis for Approach: '{approach.upper()}' ---")
        df_approach = df_raw[df_raw['Approach'] == approach].copy()
        df_complete = df_approach.dropna(subset=available_judge_cols)

        if len(df_complete) < 2:
            print(f"Not enough complete samples to calculate Kappa. Skipping.")
            continue

        for col in available_judge_cols:
            df_complete.loc[:, col] = df_complete[col].astype(int)

        judge_pairs = combinations(available_judge_cols, 2)
        kappa_results = {}
        for j1_col, j2_col in judge_pairs:
            kappa_val = cohen_kappa_score(df_complete[j1_col], df_complete[j2_col], weights='quadratic')
            j1_name, j2_name = j1_col.replace('_Score', ''), j2_col.replace('_Score', '')
            kappa_results[(j1_name, j2_name)] = kappa_val

        print("Weighted Cohen's Kappa (Agreement beyond chance):")
        for (j1, j2), val in kappa_results.items():
            print(f"- {j1} vs. {j2}: Kappa = {val:.3f}")

        judge_names = [col.replace('_Score', '') for col in available_judge_cols]
        kappa_matrix = pd.DataFrame(np.nan, index=judge_names, columns=judge_names)
        for (j1, j2), val in kappa_results.items():
            kappa_matrix.loc[j1, j2] = val
            kappa_matrix.loc[j2, j1] = val
        np.fill_diagonal(kappa_matrix.values, 1.0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(kappa_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=.5, vmin=0, vmax=1)
        plt.title(f"Inter-Judge Agreement (Weighted Kappa)\nApproach: '{approach.upper()}'", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f"judge_kappa_{approach}.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved Kappa heatmap: {plot_filename}")
        plt.close()

# === Module: Approach & Improvement Analysis ===

def plot_direct_vs_target_scores(df_all_raw, output_dir):
    """
    Generates a bar plot comparing 'direct' vs 'target' mean scores for each model.
    """
    print("\n--- Running: Direct vs. Target Score Comparison Plot ---")
    if df_all_raw.empty:
        print("Raw DataFrame is empty. Skipping plot.")
        return

    df_filtered = df_all_raw[df_all_raw['Approach'].isin(['direct', 'target'])]
    if df_filtered.empty:
        print("No data found for 'direct' or 'target' approaches.")
        return

    mean_scores = df_filtered.groupby(['Model', 'Approach'])['Score'].mean().unstack()
    if 'direct' not in mean_scores.columns: mean_scores['direct'] = pd.NA
    if 'target' not in mean_scores.columns: mean_scores['target'] = pd.NA

    mean_scores_plot = mean_scores[['direct', 'target']].dropna(how='all')
    if mean_scores_plot.empty:
        print("Not enough data to plot direct vs target comparison.")
        return

    os.makedirs(output_dir, exist_ok=True)
    mean_scores_plot.plot(kind='bar', figsize=(14, 8))
    plt.title('Comparison of Mean Scores: Direct vs Target RAG by Model', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean LLM Judge Score', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, "direct_vs_target_scores.png")
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.close()

def plot_model_improvement_by_year(df_processed, output_dir):
    """
    For each model, plots the mean improvement of various approaches over 'direct', by year.
    """
    print("\n--- Running: Model Improvement by Year Plot ---")
    if df_processed.empty:
        print("Processed DataFrame is empty. Skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    models = df_processed['Model'].unique()

    for model_name in models:
        df_model = df_processed[df_processed['Model'] == model_name].copy()
        improvement_cols = [c for c in df_model.columns if '_improvement' in c and df_model[c].notna().any()]
        if not improvement_cols:
            continue

        df_model['Year'] = pd.to_numeric(df_model['Year'], errors='coerce')
        yearly_improvement = df_model.groupby('Year')[improvement_cols].mean().sort_index()
        
        trendline_cols = [c for c in ['k5_improvement', 'k10_improvement', 'target_improvement'] if c in yearly_improvement.columns]
        if trendline_cols:
            yearly_improvement['mean_trendline'] = yearly_improvement[trendline_cols].mean(axis=1)

        if yearly_improvement.empty:
            continue

        plt.figure(figsize=(12, 7))
        for col in improvement_cols:
            label = col.replace('_improvement', '').upper()
            sns.lineplot(data=yearly_improvement, x=yearly_improvement.index, y=col, label=label, marker='o')
        
        if 'mean_trendline' in yearly_improvement.columns:
            sns.lineplot(data=yearly_improvement, x=yearly_improvement.index, y='mean_trendline',
                         label='Mean Trendline', marker='s', linewidth=2, linestyle='--', color='red')

        plt.title(f'Mean Improvement Over Direct by Year - Model: {model_name}', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Improvement in Score (vs Direct)', fontsize=12)
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='No Improvement')
        plt.legend(title='Approach')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        plot_filename = os.path.join(output_dir, f"improvement_by_year_{model_name}.png")
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

def plot_score_distributions(df_raw, output_dir):
    """
    Generates box plots of score distributions for each model and approach.
    """
    print("\n--- Running: Score Distribution Plots ---")
    if df_raw.empty:
        print("Raw DataFrame empty. Skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    models = sorted(df_raw['Model'].unique())
    approach_order = ['direct', 'k5', 'k10', 'target', 'target_negated']

    for model_name in models:
        df_model_raw = df_raw[df_raw['Model'] == model_name]
        ordered_approaches = [app for app in approach_order if app in df_model_raw['Approach'].unique()]
        if df_model_raw.empty or not ordered_approaches:
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Approach", y="Score", data=df_model_raw, order=ordered_approaches, palette="Set2", showfliers=False)
        sns.stripplot(x="Approach", y="Score", data=df_model_raw, order=ordered_approaches, color=".25", alpha=0.5, size=4, jitter=True)
        plt.title(f'Score Distribution by Approach - Model: {model_name}', fontsize=16)
        plt.xlabel('Approach', fontsize=12)
        plt.ylabel('Mean LLM Judge Score', fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        plot_filename = os.path.join(output_dir, f"score_distribution_{model_name}.png")
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

# === Module: Detailed Statistical Analyses ===

def overall_performance_summary(df_raw, df_processed):
    """
    Prints summary tables of mean scores and mean improvements over 'direct'.
    """
    print("\n" + "="*20 + " Overall Performance Summary " + "="*20)
    if not df_raw.empty:
        print("\n📊 Overall Mean Scores by Model and Approach:")
        mean_scores_summary = df_raw.groupby(['Model', 'Approach'])['Score'].agg(['mean', 'std', 'count']).unstack(level='Approach')
        print(mean_scores_summary.to_string())
    else:
        print("Raw data is empty, cannot show mean scores summary.")

    if not df_processed.empty:
        print("\n📈 Mean Improvement Over Direct by Model:")
        improvement_cols = [col for col in df_processed.columns if '_improvement' in col]
        if improvement_cols:
            summary_improvements = df_processed.groupby('Model')[improvement_cols].agg(['mean', 'std', 'count'])
            summary_improvements.columns = ["_".join(col).replace('_improvement', ' Impr.') for col in summary_improvements.columns.values]
            print(summary_improvements.to_string())
        else:
            print("No improvement columns found in processed data.")
    else:
        print("Processed data is empty, cannot show mean improvements summary.")
    print("="*60)

def perform_paired_t_tests_direct_vs_target(df_processed, alpha=0.05):
    """
    Performs a paired t-test for each model comparing 'target' vs 'direct' scores.
    """
    print("\n" + "="*20 + " Paired T-test: Target vs Direct Scores " + "="*20)
    if df_processed.empty:
        print("Processed DataFrame is empty. Skipping t-tests.")
        return

    results = []
    for model_name in sorted(df_processed['Model'].unique()):
        df_model = df_processed[df_processed['Model'] == model_name]
        if 'direct' not in df_model.columns or 'target' not in df_model.columns:
            continue

        paired_data = df_model[['direct', 'target']].dropna()
        if len(paired_data) < MIN_SAMPLES_FOR_TTEST:
            continue

        t_stat, p_value = ttest_rel(paired_data['target'], paired_data['direct'])
        is_sig = (p_value < alpha) and (t_stat > 0)
        results.append({
            'Model': model_name, 'N_pairs': len(paired_data),
            'Mean Direct': f"{paired_data['direct'].mean():.3f}",
            'Mean Target': f"{paired_data['target'].mean():.3f}",
            'T-statistic': f"{t_stat:.3f}", 'P-value': f"{p_value:.4f}",
            f'Significant Improvement (p<{alpha})': "Yes" if is_sig else "No"
        })

    if results:
        print(pd.DataFrame(results).set_index('Model').to_string())
    else:
        print("No models had sufficient data for t-tests.")
    print("="*60)

def perform_cross_model_t_tests(df_processed, alpha=0.05):
    """
    Performs paired t-tests comparing MedGemma vs Gemma scores for 'target' and 'direct'.
    """
    print("\n" + "="*20 + " Cross-Model Paired T-tests: MedGemma vs Gemma " + "="*20)
    if df_processed.empty:
        print("Processed DataFrame is empty. Cannot perform t-tests.")
        return

    available_models = df_processed['Model'].unique()
    medgemma_models = [m for m in available_models if 'medgemma' in m.lower()]
    gemma_models = [m for m in available_models if 'gemma' in m.lower() and 'medgemma' not in m.lower()]

    if not medgemma_models or not gemma_models:
        print("Could not find both Gemma and MedGemma models in the data. Skipping.")
        return

    all_results = []
    for score_type in ['target', 'direct']:
        if score_type not in df_processed.columns:
            continue
        for medgemma_model in medgemma_models:
            for gemma_model in gemma_models:
                medgemma_data = df_processed[df_processed['Model'] == medgemma_model]
                gemma_data = df_processed[df_processed['Model'] == gemma_model]

                merged_data = pd.merge(
                    medgemma_data[['Number', score_type]].rename(columns={score_type: 'score_medgemma'}),
                    gemma_data[['Number', score_type]].rename(columns={score_type: 'score_gemma'}),
                    on='Number', how='inner'
                )
                paired_data = merged_data.dropna()

                if len(paired_data) < MIN_SAMPLES_FOR_TTEST:
                    continue

                medgemma_scores = paired_data['score_medgemma']
                gemma_scores = paired_data['score_gemma']
                t_stat, p_value = ttest_rel(medgemma_scores, gemma_scores)

                is_significant = p_value < alpha
                significance_text = "No"
                if is_significant:
                    significance_text = "Yes (MedGemma > Gemma)" if t_stat > 0 else "Yes (MedGemma < Gemma)"

                all_results.append({
                    'Score_Type': score_type.capitalize(),
                    'Comparison': f"{medgemma_model} vs {gemma_model}",
                    'N_pairs': len(paired_data),
                    'Mean_MedGemma': f"{medgemma_scores.mean():.3f}",
                    'Mean_Gemma': f"{gemma_scores.mean():.3f}",
                    'T_statistic': f"{t_stat:.3f}", 'P_value': f"{p_value:.4f}",
                    f'Significant_p<{alpha}': significance_text
                })

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
    else:
        print("No cross-model comparisons could be performed due to insufficient data.")
    print("="*80)

def correlation_with_suitability(df_processed):
    """
    Analyzes correlation between Suitability Score and RAG improvements.
    """
    print("\n" + "="*20 + " Correlation with Suitability Score " + "="*20)
    if df_processed.empty or 'Suitability Score' not in df_processed.columns:
        print("Processed DataFrame empty or 'Suitability Score' missing.")
        return

    all_correlations = []
    for model_name in sorted(df_processed['Model'].unique()):
        df_model_data = df_processed[df_processed['Model'] == model_name].copy()
        df_model_data['Suitability Score'] = pd.to_numeric(df_model_data['Suitability Score'], errors='coerce')
        imp_cols = [c for c in ['k5_improvement', 'k10_improvement', 'target_improvement'] if c in df_model_data.columns]
        if not imp_cols or df_model_data['Suitability Score'].isnull().all():
            continue

        corr_data = df_model_data[['Suitability Score'] + imp_cols].dropna()
        if len(corr_data) < 2:
            continue

        corr_matrix = corr_data.corr()
        model_corrs = corr_matrix.loc['Suitability Score', imp_cols].to_frame().T
        model_corrs.index = [model_name]
        all_correlations.append(model_corrs)

    if all_correlations:
        all_corrs_df = pd.concat(all_correlations)
        all_corrs_df.columns = [c.replace('_improvement', ' Impr.') for c in all_corrs_df.columns]
        print(all_corrs_df.to_string())
    else:
        print("Could not calculate suitability correlations.")
    print("="*60)

def topic_level_analysis(df_processed, output_dir):
    """
    Analyzes which topics benefit most/least from RAG, per model.
    """
    print("\n" + "="*20 + " Topic-Level Average Improvements " + "="*20)
    if df_processed.empty or 'Topic' not in df_processed.columns:
        print("Processed DataFrame empty or 'Topic' column missing.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for model_name in sorted(df_processed['Model'].unique()):
        print(f"\n--- Model: {model_name} ---")
        df_model_data = df_processed[df_processed['Model'] == model_name].copy()
        if 'target_improvement' in df_model_data.columns and df_model_data['target_improvement'].notna().any():
            topic_improvements = df_model_data.groupby('Topic')['target_improvement'].agg(['mean', 'count', 'std']).sort_values(by='mean', ascending=False)
            if topic_improvements.empty:
                continue

            print(topic_improvements.to_string())
            if len(topic_improvements) > 1 and len(topic_improvements) < 50:
                plt.figure(figsize=(10, max(6, len(topic_improvements) * 0.4)))
                topic_improvements['mean'].plot(kind='barh')
                plt.title(f'Avg. Target Improvement by Topic - Model: {model_name}')
                plt.xlabel('Mean Improvement Score')
                plt.ylabel('Topic')
                plt.grid(axis='x', linestyle=':')
                plt.tight_layout()
                plot_filename = os.path.join(output_dir, f"topic_improvement_{model_name}.png")
                plt.savefig(plot_filename)
                print(f"Saved plot: {plot_filename}")
                plt.close()
    print("="*60)

# === Module: BERTScore Analysis ===

def get_bertscore_model():
    """Loads the BERTScore model instance once and stores it globally."""
    global BERTSCORE_MODEL_INSTANCE
    if BERTSCORE_MODEL_INSTANCE is None:
        print(f"Loading BERTScore model ({BERTSCORE_MODEL_TYPE})... This may take a moment.")
        try:
            BERTSCORE_MODEL_INSTANCE = hf_load("bertscore")
            print("BERTScore model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Error loading BERTScore model: {e}. Analyses will be skipped.")
            BERTSCORE_MODEL_INSTANCE = "ERROR" # Mark as failed

    return None if BERTSCORE_MODEL_INSTANCE == "ERROR" else BERTSCORE_MODEL_INSTANCE

def calculate_and_summarize_bertscore(df_raw):
    """
    Calculates BERTScore for each Model/Approach group.
    """
    print("\n" + "="*20 + " BERTScore Analysis " + "="*20)
    bs_metric_loader = get_bertscore_model()
    if not bs_metric_loader:
        return pd.DataFrame()

    if 'Reference_Conclusion' not in df_raw.columns or 'Predicted_Conclusion' not in df_raw.columns:
        print("Warning: Conclusion columns not found. Skipping BERTScore.")
        return pd.DataFrame()

    df_bert_ready = df_raw.dropna(subset=['Reference_Conclusion', 'Predicted_Conclusion']).copy()
    if df_bert_ready.empty:
        print("No valid conclusion pairs found to calculate BERTScore.")
        return pd.DataFrame()

    results_list = []
    grouped = df_bert_ready.groupby(['Model', 'Approach'])
    
    for (model, approach), group in tqdm(grouped, desc="BERTScore Progress"):
        if len(group) < MIN_SAMPLES_FOR_BERTSCORE:
            continue
        
        refs = group['Reference_Conclusion'].tolist()
        preds = group['Predicted_Conclusion'].tolist()
        
        try:
            compute_kwargs = {"predictions": preds, "references": refs, "lang": BERTSCORE_LANG, "model_type": BERTSCORE_MODEL_TYPE}
            if BERTSCORE_NUM_LAYERS is not None:
                compute_kwargs["num_layers"] = BERTSCORE_NUM_LAYERS
            
            bs_results = bs_metric_loader.compute(**compute_kwargs)
            results_list.append({
                'Model': model, 'Approach': approach, 'N_Pairs': len(refs),
                'BERT_F1': np.mean(bs_results['f1']),
                'BERT_Precision': np.mean(bs_results['precision']),
                'BERT_Recall': np.mean(bs_results['recall']),
            })
        except Exception as e:
            print(f"\nError calculating BERTScore for {model}-{approach}: {e}")

    if not results_list:
        print("No BERTScore results were computed.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(results_list)
    print("\n📊 BERTScore Summary (Mean Scores per Model/Approach):")
    print(summary_df.to_string(float_format="%.4f"))
    print("="*60)
    return summary_df

def plot_bertscore_direct_vs_target(df_bertscore_summary, output_dir):
    """
    Generates a bar plot comparing 'direct' vs 'target' mean BERTScore F1.
    """
    print("\n--- Running: BERTScore F1 Comparison Plot ---")
    if df_bertscore_summary.empty:
        print("BERTScore summary is empty. Skipping plot.")
        return

    df_plot = df_bertscore_summary[df_bertscore_summary['Approach'].isin(['direct', 'target', 'target_negated'])].copy()
    if df_plot.empty:
        print("No data for 'direct' or 'target' approaches with valid BERT_F1 scores.")
        return

    pivot_plot = df_plot.pivot(index='Model', columns='Approach', values='BERT_F1').sort_index()
    if pivot_plot.empty:
        print("Not enough data to plot BERTScore F1 comparison.")
        return

    os.makedirs(output_dir, exist_ok=True)
    pivot_plot.plot(kind='bar', figsize=(14, 8), colormap='Paired')
    plt.title('Comparison of Mean BERTScore F1 by Model and Approach', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean BERTScore F1', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Approach')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, "bertscore_f1_comparison.png")
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.close()


# === Module: Special Preprocessing Utilities ===

def preprocess_noriz_data(output_dir):
    """
    Specialized script to process 'filtered_evaluations_noriz.csv' and 'tasks.json'.
    This is a one-off utility and not part of the main analysis pipeline.
    """
    print("\n--- Running: Noriz Data Preprocessing Utility ---")
    try:
        df = pd.read_csv('filtered_evaluations_noriz.csv')
        with open('tasks.json', 'r') as f:
            tasks_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}. Skipping this step.")
        return

    tasks = tasks_data.get('evaluationTasks', [])
    task_mapping = {task['taskId']: task for task in tasks}

    def get_model_info(task_id, key):
        task = task_mapping.get(task_id, {})
        return task.get('modelIdentities', {}).get(key)
        
    df['modelA'] = df['task_id'].apply(lambda x: get_model_info(x, 'modelA'))
    df['modelB'] = df['task_id'].apply(lambda x: get_model_info(x, 'modelB'))

    def get_comparison_type(row):
        models = {row['modelA'], row['modelB']}
        if models == {'MedGemma Target Conclusion', 'MedGemma Direct Conclusion'}:
            return 'MedGemma Target vs MedGemma Direct'
        if models == {'Gemma Target Conclusion', 'Gemma Direct Conclusion'}:
            return 'Gemma Target vs Gemma Direct'
        if models == {'MedGemma Target Conclusion', 'Gemma Target Conclusion'}:
            return 'MedGemma Target vs Gemma Target'
        if models == {'MedGemma Direct Conclusion', 'Gemma Direct Conclusion'}:
            return 'MedGemma Direct vs Gemma Direct'
        return 'Other'

    df['comparison_type'] = df.apply(get_comparison_type, axis=1)

    print("\nBreakdown by comparison type:")
    print(df['comparison_type'].value_counts())

    os.makedirs(output_dir, exist_ok=True)
    for comp_type, data in df.groupby('comparison_type'):
        if comp_type != 'Other':
            filename = comp_type.lower().replace(' ', '_').replace('__', '_') + ".csv"
            output_path = os.path.join(output_dir, filename)
            data.to_csv(output_path, index=False)
            print(f"Saved preprocessed file: {output_path}")


# --- 5. MAIN EXECUTION (CLI) ---

def main_cli():
    """
    Main function to parse command-line arguments and run selected analyses.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis modules on LLM evaluation data.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- General Arguments ---
    parser.add_argument(
        '--data-dir', type=str, default=".",
        help="Directory containing the input CSV files (default: current directory)."
    )
    parser.add_argument(
        '--output-dir', type=str, default="analysis_output",
        help="Directory to save plots and output files (default: 'analysis_output')."
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help="Significance level (alpha) for statistical tests (default: 0.05)."
    )
    
    # --- Analysis Module Selection ---
    analysis_group = parser.add_argument_group('Analysis Modules', 'Select one or more analyses to run.')
    analysis_group.add_argument('--all', action='store_true', help='Run all standard analyses below.')
    analysis_group.add_argument('--summary', action='store_true', help='Print overall performance tables.')
    analysis_group.add_argument('--compare-judges', action='store_true', help="Plot overlapped judge scores.")
    analysis_group.add_argument('--similarity', action='store_true', help="Run judge similarity (Spearman correlation).")
    analysis_group.add_argument('--kappa', action='store_true', help="Run judge agreement (Cohen's Kappa).")
    analysis_group.add_argument('--direct-vs-target', action='store_true', help="Plot direct vs. target scores.")
    analysis_group.add_argument('--improvement-by-year', action='store_true', help="Plot score improvement by year.")
    analysis_group.add_argument('--distributions', action='store_true', help="Plot score distributions.")
    analysis_group.add_argument('--ttest-target', action='store_true', help="Run paired t-tests (Target vs Direct).")
    analysis_group.add_argument('--ttest-cross-model', action='store_true', help="Run cross-model t-tests (Gemma vs MedGemma).")
    analysis_group.add_argument('--suitability-corr', action='store_true', help="Run correlation with suitability score.")
    analysis_group.add_argument('--topic-analysis', action='store_true', help="Run topic-level improvement analysis.")
    analysis_group.add_argument('--bertscore', action='store_true', help='Calculate and plot BERTScore.')

    # --- Utility Scripts ---
    util_group = parser.add_argument_group('Utility Scripts', 'Run standalone utility tasks.')
    util_group.add_argument('--preprocess-noriz', action='store_true', help='Run the special Noriz data preprocessing script.')

    args = parser.parse_args()
    
    # Check if any action was requested
    is_any_analysis_selected = any([
        args.all, args.summary, args.compare_judges, args.similarity, args.kappa,
        args.direct_vs_target, args.improvement_by_year, args.distributions,
        args.ttest_target, args.ttest_cross_model, args.suitability_corr,
        args.topic_analysis, args.bertscore, args.preprocess_noriz
    ])
    if not is_any_analysis_selected:
        parser.print_help()
        print("\nError: No analysis module selected. Please specify at least one module to run (e.g., --summary) or use --all.")
        return

    # --- Setup ---
    sns.set_theme(style="whitegrid")
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)
    
    # --- Run Utility Scripts ---
    if args.preprocess_noriz:
        preprocess_noriz_data(output_dir=os.path.join(output_base_dir, "preprocessed_noriz"))

    # --- Run Main Analysis Pipeline ---
    needs_data_loading = any([
        args.all, args.summary, args.compare_judges, args.similarity, args.kappa,
        args.direct_vs_target, args.improvement_by_year, args.distributions,
        args.ttest_target, args.ttest_cross_model, args.suitability_corr,
        args.topic_analysis, args.bertscore
    ])
    
    if not needs_data_loading:
        print("\nFinished running selected utilities.")
        return
        
    print(f"\n--- Loading data from directory: {args.data_dir} ---")
    df_raw = load_all_data(directory=args.data_dir)
    if df_raw.empty:
        print("Fatal: No data was loaded. Cannot proceed with analyses.")
        return

    # Process data only if needed by a selected module
    needs_processed_data = any([
        args.all, args.summary, args.improvement_by_year, args.ttest_target,
        args.ttest_cross_model, args.suitability_corr, args.topic_analysis
    ])
    df_processed = pd.DataFrame()
    if needs_processed_data:
        print("--- Processing data for wide-format analysis ---")
        df_processed = process_data_for_analysis(df_raw)
        if df_processed.empty:
            print("Warning: Processed data is empty. Analyses depending on it may fail or be skipped.")

    # Execute selected modules
    if args.all or args.summary:
        overall_performance_summary(df_raw, df_processed)
    if args.all or args.compare_judges:
        plot_model_comparison_overlapped(df_raw, output_dir=os.path.join(output_base_dir, "judge_comparison_plots"))
    if args.all or args.similarity:
        analyze_and_plot_similarity(df_raw, output_dir=os.path.join(output_base_dir, "judge_similarity_plots"))
    if args.all or args.kappa:
        analyze_and_plot_kappa(df_raw, output_dir=os.path.join(output_base_dir, "judge_kappa_plots"))
    if args.all or args.direct_vs_target:
        plot_direct_vs_target_scores(df_raw, output_dir=output_base_dir)
    if args.all or args.improvement_by_year:
        plot_model_improvement_by_year(df_processed, output_dir=output_base_dir)
    if args.all or args.distributions:
        plot_score_distributions(df_raw, output_dir=output_base_dir)
    if args.all or args.ttest_target:
        perform_paired_t_tests_direct_vs_target(df_processed, alpha=args.alpha)
    if args.all or args.ttest_cross_model:
        perform_cross_model_t_tests(df_processed, alpha=args.alpha)
    if args.all or args.suitability_corr:
        correlation_with_suitability(df_processed)
    if args.all or args.topic_analysis:
        topic_level_analysis(df_processed, output_dir=output_base_dir)
    if args.all or args.bertscore:
        df_bertscore_summary = calculate_and_summarize_bertscore(df_raw)
        if not df_bertscore_summary.empty:
            plot_bertscore_direct_vs_target(df_bertscore_summary, output_dir=output_base_dir)

    print("\n--- All selected analyses finished. ---")


if __name__ == '__main__':
    main_cli()
