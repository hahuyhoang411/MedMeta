import pandas as pd
import numpy as np
import re # Import regex for potentially more robust parsing later, though split might be enough

# Load the dataframes
try:
    # Make sure this CSV has the 'Topic' column or adjust TOPIC_COL below
    df = pd.read_csv("meta_analysis_summary_with_topics.csv")
    print("DataFrame core loaded successfully.")
    # Ensure df_ref is loaded for reference abstract check
    df_ref = pd.read_csv("pubmed_data_final.csv")
    print("DataFrame references loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the files are in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- Parameters ---
TARGET_SAMPLES = 200
MIN_YEAR = 2018
MAX_YEAR = 2025
MAX_REFS = 25
MIN_REFS = 7
TOP_N_TOPICS = 10

# Column Names - Ensure these match your CSV headers exactly
YEAR_COL = 'Date (Year)'
REFS_COL = 'Number of Refs'
TOPIC_COL = 'Topic'
REFERENCES_COL = 'References' # <<< NEW: Column containing referenced PMIDs
PARENT_PMID_COL = 'PMID' # Assuming 'PMID' is the column for the parent article's PMID in df
REF_PMID_COL = 'PMID' # Assuming 'PMID' is the column for the reference article's PMID in df_ref
REF_TEXT_COL = 'text' # Assuming 'text' contains the abstract in df_ref

RANDOM_SEED = 42 # for reproducibility
OUTPUT_CSV = 'MedMeta_eval.csv'

# --- Input Validation (Check required columns in both DFs) ---
required_df_cols = [YEAR_COL, REFS_COL, TOPIC_COL, REFERENCES_COL, PARENT_PMID_COL]
if not all(col in df.columns for col in required_df_cols):
    missing = [col for col in required_df_cols if col not in df.columns]
    print(f"Error: Missing required columns in '{'meta_analysis_summary_with_topics.csv'}': {missing}")
    print(f"Available columns in df: {df.columns.tolist()}")
    exit()

required_df_ref_cols = [REF_PMID_COL, REF_TEXT_COL]
if not all(col in df_ref.columns for col in required_df_ref_cols):
    missing = [col for col in required_df_ref_cols if col not in df_ref.columns]
    print(f"Error: Missing required columns in '{'pubmed_data_final.csv'}': {missing}")
    print(f"Available columns in df_ref: {df_ref.columns.tolist()}")
    exit()

# --- Process df_ref to identify PMIDs with Abstracts ---
print("\n--- Processing reference data (df_ref) ---")

# Clean and convert PMID in df_ref to string for consistent lookup
df_ref[REF_PMID_COL] = df_ref[REF_PMID_COL].astype(str) # Convert to string first
# Replace potential NaN strings or empty strings in PMID with actual NaN
df_ref[REF_PMID_COL] = df_ref[REF_PMID_COL].replace(['nan', 'NaN', 'None', ''], np.nan)

# Handle text column: fill NaN with empty string for string operations
df_ref[REF_TEXT_COL] = df_ref[REF_TEXT_COL].astype(str).fillna('')

# Identify rows with valid abstracts (text is not empty and does not contain the 'No abstract found' marker)
abstract_mask = (df_ref[REF_TEXT_COL].str.strip() != '') & \
                (~df_ref[REF_TEXT_COL].str.contains('# Abstract: No abstract found', na=False)) # na=False is redundant due to fillna('') but good practice

# Get the set of PMIDs that have abstracts
# Drop NaNs from the PMID column before creating the set
pmids_with_abstracts = set(df_ref.loc[abstract_mask & df_ref[REF_PMID_COL].notna(), REF_PMID_COL].unique())

print(f"Total unique PMIDs in df_ref: {df_ref[REF_PMID_COL].nunique()}")
print(f"Unique PMIDs in df_ref with detected abstracts: {len(pmids_with_abstracts)}")
print(f"Unique PMIDs in df_ref without abstracts: {df_ref[REF_PMID_COL].nunique() - len(pmids_with_abstracts)}")


# --- Define function to check if all references have abstracts ---
def check_all_references_have_abstract(references_string, pmids_with_abstracts_set):
    """
    Checks if all PMIDs in a comma-separated string are present in the set of PMIDs with abstracts.
    Returns True if all references have abstracts (or if there are no references), False otherwise.
    Handles NaN, empty strings, and parsing errors.
    """
    if pd.isna(references_string) or references_string.strip() == '':
        # If the parent article has no references listed, it cannot be sampled based on reference abstract availability.
        # Or perhaps, if it has no references, the condition "all references have abstracts" is vacuously true?
        # Let's assume articles *must* have valid references to be eligible for this benchmark.
        return False # Exclude articles with no references listed

    # Split the string by comma and strip whitespace, filter out empty results from split like '123,'
    ref_pmid_strings = [pmid.strip() for pmid in references_string.split(',') if pmid.strip()]

    if not ref_pmid_strings:
         # If after splitting and cleaning, no valid PMID strings are found
         return False # Exclude articles if their references string is invalid or empty

    # Check if ALL parsed reference PMIDs are in the set of PMIDs with abstracts
    # We convert parsed PMIDs to string to match the type in pmids_with_abstracts_set
    all_references_have_abstract = all(pmid_str in pmids_with_abstracts_set for pmid_str in ref_pmid_strings)

    return all_references_have_abstract


# --- Data Cleaning and Initial Filtering (Apply Reference Filter Here) ---
print("\n--- Cleaning and filtering main data (df) ---")

# Convert critical columns to appropriate types, coercing errors
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors='coerce')
df[REFS_COL] = pd.to_numeric(df[REFS_COL], errors='coerce')
# Ensure topic column is treated as string/category, handle potential NaNs
df[TOPIC_COL] = df[TOPIC_COL].astype(str) # Convert to string first
df[TOPIC_COL] = df[TOPIC_COL].replace(['nan', 'NaN', 'None', ''], np.nan) # Replace common string NaNs
# Ensure references column is string, handle potential NaNs gracefully in the check function
df[REFERENCES_COL] = df[REFERENCES_COL].astype(str) # Convert to string first, NaNs will remain unless filled

# Drop rows where conversion failed or values are missing in critical columns (except References for now, handled in check_function)
critical_cols_for_initial_dropna = [YEAR_COL, REFS_COL, TOPIC_COL, PARENT_PMID_COL] # Added PARENT_PMID_COL
df_cleaned = df.dropna(subset=critical_cols_for_initial_dropna).copy()
# Ensure year is integer after cleaning
df_cleaned[YEAR_COL] = df_cleaned[YEAR_COL].astype(int)

print(f"\nOriginal DataFrame shape: {df.shape}")
print(f"DataFrame shape after cleaning NaNs in critical columns: {df_cleaned.shape}")

# --- NEW: Apply Reference Abstract Filter ---
print("\n--- Applying filter: All references must have abstracts ---")
# Apply the check function to the REFERENCES_COL
# Using .apply() might be slow for very large datasets, but it's robust for parsing strings.
# Wrap in try/except in case of unexpected errors during apply
try:
    has_valid_references_mask = df_cleaned[REFERENCES_COL].apply(
        lambda x: check_all_references_have_abstract(x, pmids_with_abstracts)
    )
    df_filtered_by_references = df_cleaned[has_valid_references_mask].copy()
    print(f"Number of rows remaining after reference abstract filter: {len(df_filtered_by_references)}")
except Exception as e:
    print(f"Error applying reference abstract filter: {e}")
    # Decide how to handle: either exit or continue without this filter (not recommended)
    print("Exiting due to reference filter error.")
    exit()


# 1. Filter the DataFrame based on YEAR and REFS conditions (Apply to the reference-filtered DF)
eligible_df = df_filtered_by_references[ # <<< Apply filter to the result of reference filtering
    (df_filtered_by_references[YEAR_COL] >= MIN_YEAR) &
    (df_filtered_by_references[YEAR_COL] <= MAX_YEAR) &
    (df_filtered_by_references[REFS_COL] < MAX_REFS) &
    (df_filtered_by_references[REFS_COL] > MIN_REFS)
].copy()

print(f"\nNumber of rows eligible after Year/Refs filtering (Years {MIN_YEAR}-{MAX_YEAR}, Refs {MIN_REFS+1}-{MAX_REFS-1}): {len(eligible_df)}")

# --- Filter by Top N Topics (Apply to the eligible_df) ---
if not eligible_df.empty:
    # Calculate topic frequencies within the eligible data
    topic_counts = eligible_df[TOPIC_COL].value_counts()

    # Determine the actual number of top topics to keep (handles cases with < TOP_N_TOPICS unique topics)
    # Ensure we don't try to get more topics than available
    n_topics_to_keep = min(TOP_N_TOPICS, len(topic_counts[topic_counts > 0])) # Only count topics with > 0 instances

    if n_topics_to_keep > 0:
        # Get the names of the top N topics
        top_topics = topic_counts.nlargest(n_topics_to_keep).index.tolist()
        print(f"\nIdentified Top {n_topics_to_keep} topics (out of {len(topic_counts)} unique topics in eligible data):")
        print(topic_counts.head(n_topics_to_keep)) # Show counts of top topics

        # Filter the eligible_df to keep only rows with these top topics
        eligible_topic_df = eligible_df[eligible_df[TOPIC_COL].isin(top_topics)].copy()
        print(f"Number of rows after filtering for Top {n_topics_to_keep} Topics: {len(eligible_topic_df)}")
    else:
        print(f"\nWarning: No topics with counts > 0 found in the data eligible by year/refs/references. Cannot filter by topic.")
        eligible_topic_df = pd.DataFrame(columns=eligible_df.columns) # Empty DF
else:
    print("\nNo rows eligible after reference/year/ref filtering. Cannot proceed to topic filtering or sampling.")
    eligible_topic_df = pd.DataFrame(columns=df_cleaned.columns) # Empty DF


# --- Sampling Logic (Stratified by YEAR and TOPIC) ---
final_samples_df = pd.DataFrame() # Initialize an empty DataFrame

# Proceed only if we have data after all filtering steps
if eligible_topic_df.empty:
    print("\nNo rows match the specified criteria (References Abstracts, Year, Refs, Top Topics). Cannot sample.")
elif len(eligible_topic_df) <= TARGET_SAMPLES:
    print(f"\nFound {len(eligible_topic_df)} eligible rows (after all filters), which is less than or equal to the target {TARGET_SAMPLES}.")
    print("Returning all eligible rows.")
    final_samples_df = eligible_topic_df.copy()
else:
    print(f"\nAttempting to sample {TARGET_SAMPLES} rows from {len(eligible_topic_df)} eligible rows (filtered by References Abstracts, Year, Refs, Top Topics)...")
    # 2. Stratified Sampling: Group by YEAR and TOPIC
    grouped = eligible_topic_df.groupby([YEAR_COL, TOPIC_COL])
    n_groups = grouped.ngroups # Number of unique (Year, Topic) combinations found

    if n_groups == 0:
         # This case should ideally not be reached if eligible_topic_df is not empty,
         # but adding a check is safe.
         print("Error: No groups found for stratified sampling despite eligible data existing.")
         print("This might happen if the grouping columns have unexpected values or NaNs.")
         exit()

    # Calculate base samples per group and the remainder
    base_samples_per_group = TARGET_SAMPLES // n_groups
    remainder = TARGET_SAMPLES % n_groups

    sampled_indices = []
    # potential_extra_indices = {} # Not strictly needed if sampling from flat list

    print(f"Found {n_groups} unique ({YEAR_COL}, {TOPIC_COL}) combinations in the eligible data.")
    print(f"Aiming for base samples per ({YEAR_COL}, {TOPIC_COL}) group: {base_samples_per_group}")
    print(f"Remainder samples to distribute: {remainder}")

    # Set seed for reproducibility before any sampling choice
    np.random.seed(RANDOM_SEED)
    random_state = np.random.get_state() # Save the state to restore later if needed

    # Use grouped.sample() which is more idiomatic for stratified sampling
    # We sample 'base_samples_per_group' or the group size, whichever is smaller
    # Need to handle remainder distribution manually or use a more complex sample approach if sizes are very uneven
    # Let's stick to the current logic of base + remainder pool for clarity and control

    # Alternative approach using grouped.sample for base samples
    base_samples_list = []
    potential_extras_list = [] # List of indices for potential extra samples

    # Ensure we don't try to sample more than available in any group for base
    # group.sample(n=...) will raise an error if n > group.size.
    # We can use frac=... or min(n, group.size)
    # Let's use sample(n=min(group.size, base_samples_per_group), replace=False) inside the loop

    for (year, topic), group_df in grouped:
        n_group = len(group_df)
        n_to_sample_base = min(n_group, base_samples_per_group)

        if n_to_sample_base > 0:
             # Sample base indices from this group
             sampled_base_group_indices = group_df.sample(n=n_to_sample_base, replace=False, random_state=RANDOM_SEED).index.tolist()
             base_samples_list.extend(sampled_base_group_indices)

             # Identify potential extras from this group (indices not sampled in base)
             remaining_in_group_indices = group_df.index.difference(sampled_base_group_indices).tolist()
             potential_extras_list.extend(remaining_in_group_indices)
        else:
             # If base_samples_per_group is 0 or group is too small, add all group indices to potential extras
             potential_extras_list.extend(group_df.index.tolist())

    sampled_indices = base_samples_list
    num_currently_sampled = len(sampled_indices)
    num_needed = TARGET_SAMPLES - num_currently_sampled

    print(f"Collected {num_currently_sampled} samples in the initial base pass.")
    print(f"Need {num_needed} more samples for the remainder.")
    print(f"Pool of potential extra samples contains {len(potential_extras_list)} indices.")

    if num_needed > 0 and potential_extras_list:
        # Determine how many extras we can actually take
        n_extras_to_take = min(num_needed, len(potential_extras_list))
        print(f"Sampling {n_extras_to_take} additional 'remainder' samples from the pool...")

        # Sample from the pool of potential extras (using same random state for reproducibility)
        np.random.set_state(random_state) # Restore state to ensure remainder sampling is reproducible relative to base
        extra_indices_chosen = np.random.choice(potential_extras_list, size=n_extras_to_take, replace=False).tolist()
        sampled_indices.extend(extra_indices_chosen)

    elif num_needed > 0:
        print(f"Warning: Needed {num_needed} more samples for the remainder, but the pool of potential extras was empty.")


    # 5. Retrieve the final sampled DataFrame using the collected indices
    # Sample from the eligible_topic_df which already has all filters applied
    if sampled_indices: # Only access .loc if we actually sampled something
        final_samples_df = eligible_topic_df.loc[sampled_indices].copy()
    else:
        final_samples_df = pd.DataFrame(columns=eligible_topic_df.columns) # Return empty if no samples

    # Optional: Shuffle the final result so it's not ordered by year/topic groups
    if not final_samples_df.empty:
         final_samples_df = final_samples_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# --- Output ---
print(f"\n--- Sampling Complete ---")
print(f"Target number of samples: {TARGET_SAMPLES}")
print(f"Actual number of samples obtained: {len(final_samples_df)}")

if not final_samples_df.empty:
    print("\nDistribution of final samples by Year:")
    print(final_samples_df[YEAR_COL].value_counts().sort_index())

    print(f"\nDistribution of final samples by Topic (Top {min(TOP_N_TOPICS, final_samples_df[TOPIC_COL].nunique())}):")
    # Show counts for the topics actually present in the final sample
    print(final_samples_df[TOPIC_COL].value_counts())

    # Optional: Verify distribution across Year and Topic combined
    # print("\nDistribution by Year and Topic combination:")
    # print(final_samples_df.groupby([YEAR_COL, TOPIC_COL]).size().unstack(fill_value=0))

    print("\nFinal Sampled DataFrame Info:")
    final_samples_df.info()

    try:
        final_samples_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully saved {len(final_samples_df)} samples to '{OUTPUT_CSV}'")
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")

else:
    print("\nNo samples were generated based on the specified criteria and data.")

# Now 'final_samples_df' contains your desired sample set, filtered and stratified appropriately.