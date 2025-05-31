import pandas as pd
import numpy as np
import re

# --- Parameters ---
# Core parameters for the new redistribution logic and I/O
TARGET_SAMPLES = 100        # Target for the new redistribution logic (from new script's example)
RANDOM_SEED = 42            # For reproducibility of sampling

# File paths
MAIN_INPUT_CSV = "output/abstract_check_filtered_45.csv"
REF_INPUT_CSV = "pubmed_data_final.csv"
OUTPUT_CSV = 'MedMeta_eval_new_sampling.csv'         # Output for sampled parent articles
OUTPUT_REF_CSV = 'pubmed_data_reference_eval_new_sampling.csv' # Output for referenced articles

# Column Names - Ensure these match your CSV headers
# From main data (df)
YEAR_COL = 'Date (Year)'        # Original year column, used to derive 'Clean Year'
SUITABILITY_SCORE_COL = 'Suitability Score' # Required by the new sampling logic
REFERENCES_COL = 'References'   # Column containing referenced PMIDs in df
PARENT_PMID_COL = 'PMID'        # PMID of the parent article in df

# From reference data (df_ref)
REF_PMID_COL = 'PMID'           # PMID of the reference article in df_ref
REF_YEAR_COL = 'Year'           # Year of the reference article in df_ref
REF_TEXT_COL = 'text'           # Abstract text of the reference article in df_ref

# --- 0. Load Data ---
print("--- 0. Loading Data ---")
try:
    df = pd.read_csv(MAIN_INPUT_CSV)
    print(f"DataFrame core ('{MAIN_INPUT_CSV}') loaded successfully. Shape: {df.shape}")
    df_ref = pd.read_csv(REF_INPUT_CSV)
    print(f"DataFrame references ('{REF_INPUT_CSV}') loaded successfully. Shape: {df_ref.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the files are in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- Input Validation ---
print("\n--- Input Validation ---")
required_df_cols = [YEAR_COL, REFERENCES_COL, PARENT_PMID_COL, SUITABILITY_SCORE_COL]
missing_df_cols = [col for col in required_df_cols if col not in df.columns]
if missing_df_cols:
    print(f"Error: Missing required columns in '{MAIN_INPUT_CSV}': {missing_df_cols}")
    print(f"Available columns in df: {df.columns.tolist()}")
    exit()

required_df_ref_cols = [REF_PMID_COL, REF_YEAR_COL, REF_TEXT_COL]
missing_df_ref_cols = [col for col in required_df_ref_cols if col not in df_ref.columns]
if missing_df_ref_cols:
    print(f"Error: Missing required columns in '{REF_INPUT_CSV}': {missing_df_ref_cols}")
    print(f"Available columns in df_ref: {df_ref.columns.tolist()}")
    exit()
print("Input column validation passed.")

# --- A. Pre-process df_ref to identify PMIDs with Abstracts ---
print("\n--- A. Processing reference data (df_ref) to find PMIDs with abstracts ---")
df_ref[REF_PMID_COL] = df_ref[REF_PMID_COL].astype(str).replace(['nan', 'NaN', 'None', ''], np.nan)
# Convert Year in df_ref to numeric, coercing errors, then to nullable Int64
df_ref[REF_YEAR_COL] = pd.to_numeric(df_ref[REF_YEAR_COL], errors='coerce').astype('Int64')
df_ref[REF_TEXT_COL] = df_ref[REF_TEXT_COL].astype(str).fillna('')

# Identify rows with valid abstracts and valid PMIDs
abstract_mask = (df_ref[REF_TEXT_COL].str.strip() != '') & \
                (~df_ref[REF_TEXT_COL].str.contains('# Abstract: No abstract found', na=False)) & \
                (df_ref[REF_PMID_COL].notna())
pmids_with_abstracts = set(df_ref.loc[abstract_mask, REF_PMID_COL].unique())
print(f"Total unique PMIDs in df_ref: {df_ref[REF_PMID_COL].nunique()}")
print(f"Unique PMIDs in df_ref with detected abstracts: {len(pmids_with_abstracts)}")

# --- B. Define function to check if all references for an article have abstracts ---
print("\n--- B. Defining reference check function ---")
def check_all_references_have_abstract(references_string, pmids_with_abstracts_set):
    if pd.isna(references_string) or str(references_string).strip() == '':
        return False
    ref_pmid_strings = [pmid.strip() for pmid in str(references_string).split(',') if pmid.strip()]
    if not ref_pmid_strings: # No valid PMIDs found after splitting
         return False
    return all(pmid_str in pmids_with_abstracts_set for pmid_str in ref_pmid_strings)
print("Reference check function defined.")

# --- C. Apply the "all references have abstract" filter to the main DataFrame ---
print("\n--- C. Applying 'all references have abstract' filter to main DataFrame ---")
df[REFERENCES_COL] = df[REFERENCES_COL].astype(str).fillna('') # Ensure string type for apply
try:
    has_valid_references_mask = df[REFERENCES_COL].apply(
        lambda x: check_all_references_have_abstract(x, pmids_with_abstracts)
    )
    df_main_filtered = df[has_valid_references_mask].copy()
    print(f"Original df shape: {df.shape}")
    print(f"Shape of df after 'all references have abstract' filter: {df_main_filtered.shape}")

    if df_main_filtered.empty:
        print("No rows remaining after reference abstract filter. Cannot proceed. Exiting.")
        # Save empty files and exit to indicate no data met criteria
        placeholder_cols_df = df.columns if not df.empty else required_df_cols # Fallback columns
        pd.DataFrame(columns=placeholder_cols_df).to_csv(OUTPUT_CSV, index=False)
        pd.DataFrame(columns=[REF_PMID_COL, REF_YEAR_COL, REF_TEXT_COL]).to_csv(OUTPUT_REF_CSV, index=False)
        print(f"Empty '{OUTPUT_CSV}' and '{OUTPUT_REF_CSV}' created.")
        exit()
except Exception as e:
    print(f"Error applying reference abstract filter: {e}. Exiting.")
    exit()

# --- D. Apply the New Redistribution Logic ---
print("\n--- D. Applying New Redistribution Logic ---")
df_for_redistribution = df_main_filtered.copy()

# --- D.1) Clean data for redistribution (Year and Suitability Score) ---
print("--- D.1) Preparing data for new redistribution (Clean Year, Suitability Score) ---")
def extract_year(text): # From new script
    m = re.search(r'\b(19|20)\d{2}\b', str(text))
    return int(m.group()) if m else None

df_for_redistribution['Clean Year'] = df_for_redistribution[YEAR_COL].apply(extract_year)
df_for_redistribution[SUITABILITY_SCORE_COL] = pd.to_numeric(df_for_redistribution[SUITABILITY_SCORE_COL], errors='coerce')

# Drop rows if 'Clean Year' or 'Suitability Score' are NaN after conversion
initial_rows_for_redist = len(df_for_redistribution)
df_for_redistribution = df_for_redistribution.dropna(subset=['Clean Year', SUITABILITY_SCORE_COL])
df_for_redistribution['Clean Year'] = df_for_redistribution['Clean Year'].astype(int) # Ensure 'Clean Year' is int
print(f"Rows before year/suitability cleaning (input to part D): {initial_rows_for_redist}")
print(f"Rows after year/suitability cleaning (input to sampling): {len(df_for_redistribution)}")

final_samples_df = pd.DataFrame(columns=df_main_filtered.columns) # Initialize empty

if df_for_redistribution.empty:
    print("No data remains after cleaning 'Clean Year' and 'Suitability Score' for redistribution. Cannot sample.")
    # final_samples_df is already initialized as empty. Will proceed to output section.
else:
    # --- D.2) Compute per‑year quotas ---
    print("--- D.2) Computing per-year quotas ---")
    years = sorted(df_for_redistribution['Clean Year'].unique())
    k = len(years) # Number of unique years

    if k == 0: # Should be caught by df_for_redistribution.empty(), but as a safeguard
        print("No unique years found in the data for redistribution. Resulting sample will be empty.")
        # final_samples_df is already initialized as empty.
    else:
        base_quota = TARGET_SAMPLES // k
        remainder = TARGET_SAMPLES % k

        quotas = {year: base_quota for year in years}
        
        # Set seed before any random operations like choice for extra_years
        np.random.seed(RANDOM_SEED)
        
        if remainder > 0:
            # Randomly pick 'remainder' years to get +1 extra
            # Ensure we don't try to pick more years than available if remainder > k (not an issue with modulo)
            extra_years = np.random.choice(years, size=remainder, replace=False)
            for y_extra in extra_years:
                quotas[y_extra] += 1
        
        print(f"Target samples: {TARGET_SAMPLES}, Unique years for sampling: {k}")
        print("Sampling quotas by year:", {y:q for y,q in quotas.items() if q > 0}) # Print only years with non-zero quota

        # --- D.3) Do the stratified sampling by 'Clean Year' ---
        print("--- D.3) Performing stratified sampling by 'Clean Year' ---")
        samples_list = []
        
        if len(df_for_redistribution) <= TARGET_SAMPLES and sum(quotas.values()) >= len(df_for_redistribution):
            print(f"Total available data ({len(df_for_redistribution)}) is less than or equal to target quotas sum ({sum(quotas.values())}).")
            print("The sampling logic will take all items from groups smaller than their quota, and sample from larger groups.")

        for year_val, group in df_for_redistribution.groupby('Clean Year'):
            n = quotas.get(year_val, 0) # Get quota for the year, default to 0 if year not in quotas (should not happen here)
            
            if n == 0: # If quota is 0 for a year, take no samples from it
                if not group.empty: # Log if there was data but quota was zero
                    print(f"Note: Year {year_val} has {len(group)} records but its quota is 0. Skipping this year.")
                continue # Skip to next year

            if len(group) <= n:
                samp = group # Take all if group is small enough
            else:
                # Randomly sample 'n' records. random_state ensures reproducibility for this specific group's sample.
                samp = group.sample(n=n, random_state=RANDOM_SEED)
            samples_list.append(samp)

        if samples_list:
            final_samples_df = pd.concat(samples_list).reset_index(drop=True)
        else: # No samples collected (e.g. TARGET_SAMPLES = 0 or all quotas were 0 for populated groups)
            print("No samples collected during stratified sampling. Final sample set will be empty.")
            # final_samples_df remains an empty DataFrame with df_main_filtered.columns

# --- D.4) (Optional) Quick sanity check ---
if not final_samples_df.empty:
    print("\n--- D.4) Sanity Check: Counts per 'Clean Year' in final sample ---")
    print(final_samples_df['Clean Year'].value_counts().sort_index())
else:
    print("\n--- D.4) Sanity Check: Final sample is empty. ---")


# --- E. Output Parent Samples ---
print(f"\n--- E. Parent Sampling Complete & Output ---")
print(f"Target number of samples (for new redistribution logic): {TARGET_SAMPLES}")
print(f"Actual number of parent samples obtained: {len(final_samples_df)}")

if not final_samples_df.empty:
    print("\nDistribution of final parent samples by 'Clean Year':")
    print(final_samples_df['Clean Year'].value_counts().sort_index())

    if PARENT_PMID_COL in final_samples_df.columns:
         print(f"Number of unique '{PARENT_PMID_COL}' in final sample: {final_samples_df[PARENT_PMID_COL].nunique()}")
    
    # Optional: Display info about the DataFrame
    # print("\nFinal Sampled Parent DataFrame Info:")
    # final_samples_df.info(verbose=False, max_cols=5) # Show brief info

    try:
        final_samples_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully saved {len(final_samples_df)} parent samples to '{OUTPUT_CSV}'")
    except Exception as e:
        print(f"\nError saving Parent DataFrame to CSV '{OUTPUT_CSV}': {e}")
else:
    print("\nNo parent samples were generated by the new redistribution logic.")
    try: # Attempt to save an empty file with headers
        cols_to_save = df_main_filtered.columns if not df_main_filtered.empty else (df.columns if not df.empty else required_df_cols)
        pd.DataFrame(columns=cols_to_save).to_csv(OUTPUT_CSV, index=False)
        print(f"Saved an empty '{OUTPUT_CSV}' with appropriate headers.")
    except Exception as e:
        print(f"Could not save empty parent CSV '{OUTPUT_CSV}': {e}")

# --- F. Generate and Output Reference Data ---
print(f"\n--- F. Generating and Outputting Reference Data ---")
# Initialize df_ref_eval with the correct columns, even if it remains empty
df_ref_eval = pd.DataFrame(columns=[REF_PMID_COL, REF_YEAR_COL, REF_TEXT_COL])

if final_samples_df.empty:
    print("No parent samples generated. Reference data will be empty.")
else:
    if REFERENCES_COL not in final_samples_df.columns:
        print(f"Warning: Column '{REFERENCES_COL}' not found in final_samples_df. Cannot extract references.")
        all_ref_pmids_list = [] # Ensure it's an empty list
    else:
        all_ref_pmids_list = []
        # Iterate through the References column, ensure it's string, handle NaNs by filling with empty string
        for refs_str in final_samples_df[REFERENCES_COL].astype(str).fillna(''):
            # Split by comma, strip whitespace, and filter out empty strings resulting from split (e.g. '123,,456')
            pmids = [p.strip() for p in refs_str.split(',') if p.strip()]
            all_ref_pmids_list.extend(pmids)

    unique_referenced_pmids = set(all_ref_pmids_list)
    print(f"Total unique referenced PMIDs extracted from sampled parent articles: {len(unique_referenced_pmids)}")

    if unique_referenced_pmids:
        # Filter df_ref (where REF_PMID_COL is already string type from step A)
        df_ref_eval_filtered = df_ref[df_ref[REF_PMID_COL].isin(unique_referenced_pmids)].copy()
        
        # Select only the required columns for the output reference CSV
        # REF_YEAR_COL in df_ref_eval_filtered is already Int64 (allows for <NA>)
        df_ref_eval = df_ref_eval_filtered[[REF_PMID_COL, REF_YEAR_COL, REF_TEXT_COL]]
        print(f"Found {len(df_ref_eval)} corresponding reference entries in df_ref for output.")
        # Note: len(df_ref_eval) might be less than len(unique_referenced_pmids) if some PMIDs were not found in df_ref
    else:
        print("No valid referenced PMIDs to look up in df_ref.")
        # df_ref_eval remains an empty DataFrame with the correct columns

# Output Reference Data
print(f"\nActual number of reference entries for output: {len(df_ref_eval)}")
if not df_ref_eval.empty:
    # Optional: Display info about the DataFrame
    # print("\nReference Data Info:")
    # df_ref_eval.info(verbose=False, max_cols=3)
    try:
        df_ref_eval.to_csv(OUTPUT_REF_CSV, index=False)
        print(f"\nSuccessfully saved {len(df_ref_eval)} reference entries to '{OUTPUT_REF_CSV}'")
    except Exception as e:
        print(f"\nError saving Reference DataFrame to CSV '{OUTPUT_REF_CSV}': {e}")
else:
    print(f"\nNo reference data to save for '{OUTPUT_REF_CSV}'.")
    try: # Attempt to save an empty file with headers
        pd.DataFrame(columns=[REF_PMID_COL, REF_YEAR_COL, REF_TEXT_COL]).to_csv(OUTPUT_REF_CSV, index=False)
        print(f"Created an empty '{OUTPUT_REF_CSV}' file with headers.")
    except Exception as e:
         print(f"Could not create empty reference CSV '{OUTPUT_REF_CSV}': {e}")

print("\n--- Script Finished ---")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Plot: Suitability Score vs Year (scatter)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=final_samples_df, x='Clean Year', y='Suitability Score', alpha=0.6)
sns.lineplot(data=final_samples_df, x='Clean Year', y='Suitability Score', errorbar=None, estimator='mean', color='red', label='Mean')

plt.title("Suitability Score vs. Year")
plt.xlabel("Year")
plt.ylabel("Suitability Score")
plt.legend()
plt.tight_layout()
plt.show()

# Set the plot style
sns.set(style="whitegrid")

# Create the distribution plot
plt.figure(figsize=(12, 6))
sns.histplot(final_samples_df['Clean Year'], bins=30, kde=True)

# Add labels and title
plt.title("Distribution of Publication Years")
plt.xlabel("Year")
plt.ylabel("Frequency")

# Show the plot
plt.tight_layout()
plt.show()

print(len(set(final_samples_df['Topic'])))