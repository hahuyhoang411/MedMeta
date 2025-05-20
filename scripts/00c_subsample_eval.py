import pandas as pd
import numpy as np

# Load the dataframe
try:
    # Make sure this CSV has the 'Topic' column or adjust TOPIC_COL below
    df = pd.read_csv("meta_analysis_summary_with_topics.csv")
    print("DataFrame loaded successfully.")
    # Optional: Display first few rows and info to verify columns
    # print(df.head())
    # print(df.info())
except FileNotFoundError:
    print("Error: 'meta_analysis_summary_with_topics.csv' not found. Make sure the file is in the correct directory.")
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

YEAR_COL = 'Date (Year)' # Make sure this matches your exact column name
REFS_COL = 'Number of Refs' # Make sure this matches your exact column name
TOPIC_COL = 'Topic'      # <<< NEW: Specify the exact name of your topic column
# --- Ensure TOPIC_COL exists ---
if TOPIC_COL not in df.columns:
    print(f"Error: Column '{TOPIC_COL}' not found in the DataFrame.")
    print(f"Please set the 'TOPIC_COL' parameter to the correct column name.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

RANDOM_SEED = 42 # for reproducibility
OUTPUT_CSV = 'MedMeta_eval.csv'

# --- Input Validation ---
if YEAR_COL not in df.columns:
    print(f"Error: Column '{YEAR_COL}' not found in the DataFrame.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()
if REFS_COL not in df.columns:
    print(f"Error: Column '{REFS_COL}' not found in the DataFrame.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Convert year column to numeric, coercing errors (like non-year strings) to NaN
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors='coerce')
# Convert refs column to numeric, coercing errors
df[REFS_COL] = pd.to_numeric(df[REFS_COL], errors='coerce')
# Ensure topic column is treated as string/category, handle potential NaNs
df[TOPIC_COL] = df[TOPIC_COL].astype(str) # Convert to string first
df[TOPIC_COL] = df[TOPIC_COL].replace(['nan', 'NaN', 'None', ''], np.nan) # Replace common string NaNs

# Drop rows where conversion failed or values are missing in critical columns
# <<< UPDATED: Also drop rows with missing Topic >>>
critical_cols = [YEAR_COL, REFS_COL, TOPIC_COL]
df_cleaned = df.dropna(subset=critical_cols).copy()
# Ensure year is integer after cleaning
df_cleaned[YEAR_COL] = df_cleaned[YEAR_COL].astype(int)

print(f"\nOriginal DataFrame shape: {df.shape}")
print(f"DataFrame shape after cleaning NaNs in '{YEAR_COL}', '{REFS_COL}', '{TOPIC_COL}': {df_cleaned.shape}")


# 1. Filter the DataFrame based on YEAR and REFS conditions
eligible_df = df_cleaned[
    (df_cleaned[YEAR_COL] >= MIN_YEAR) &
    (df_cleaned[YEAR_COL] <= MAX_YEAR) &
    (df_cleaned[REFS_COL] < MAX_REFS) &
    (df_cleaned[REFS_COL] > MIN_REFS)
].copy()

print(f"\nNumber of rows eligible after filtering (Years {MIN_YEAR}-{MAX_YEAR}, Refs {MIN_REFS+1}-{MAX_REFS-1}): {len(eligible_df)}")

# --- NEW: Filter by Top N Topics ---
if not eligible_df.empty:
    # Calculate topic frequencies within the eligible data
    topic_counts = eligible_df[TOPIC_COL].value_counts()

    # Determine the actual number of top topics to keep (handles cases with < TOP_N_TOPICS unique topics)
    n_topics_to_keep = min(TOP_N_TOPICS, len(topic_counts))

    if n_topics_to_keep > 0:
        # Get the names of the top N topics
        top_topics = topic_counts.nlargest(n_topics_to_keep).index.tolist()
        print(f"\nIdentified Top {n_topics_to_keep} topics (out of {len(topic_counts)} unique topics in eligible data):")
        # print(top_topics) # Uncomment to see the list
        print(topic_counts.head(n_topics_to_keep)) # Show counts of top topics

        # Filter the eligible_df to keep only rows with these top topics
        eligible_topic_df = eligible_df[eligible_df[TOPIC_COL].isin(top_topics)].copy()
        print(f"Number of rows after filtering for Top {n_topics_to_keep} Topics: {len(eligible_topic_df)}")
    else:
        print(f"\nWarning: No topics found in the data eligible by year/refs. Cannot filter by topic.")
        eligible_topic_df = pd.DataFrame(columns=eligible_df.columns) # Empty DF

else:
    print("\nNo rows eligible after year/ref filtering. Cannot proceed to topic filtering or sampling.")
    eligible_topic_df = pd.DataFrame(columns=df_cleaned.columns) # Empty DF

# --- Sampling Logic (Stratified by YEAR and TOPIC) ---
final_samples_df = pd.DataFrame() # Initialize an empty DataFrame

# Proceed only if we have data after all filtering steps
if eligible_topic_df.empty:
    print("\nNo rows match the specified criteria (Year, Refs, Top Topics). Cannot sample.")
elif len(eligible_topic_df) <= TARGET_SAMPLES:
    print(f"\nFound {len(eligible_topic_df)} eligible rows (after all filters), which is less than or equal to the target {TARGET_SAMPLES}.")
    print("Returning all eligible rows.")
    final_samples_df = eligible_topic_df.copy()
else:
    print(f"\nAttempting to sample {TARGET_SAMPLES} rows from {len(eligible_topic_df)} eligible rows (filtered by Year, Refs, Top Topics)...")
    # 2. Stratified Sampling: Group by YEAR and TOPIC <<< MODIFIED
    grouped = eligible_topic_df.groupby([YEAR_COL, TOPIC_COL])
    n_groups = grouped.ngroups # Number of unique (Year, Topic) combinations found

    if n_groups == 0:
         print("Error: No groups found for stratified sampling. This shouldn't happen if eligible_topic_df is not empty.")
         exit()

    # Calculate base samples per group and the remainder
    base_samples_per_group = TARGET_SAMPLES // n_groups
    remainder = TARGET_SAMPLES % n_groups

    sampled_indices = []
    potential_extra_indices = {} # Keep track of remaining indices per group for remainder sampling

    print(f"Found {n_groups} unique (Year, Topic) combinations in the eligible data.")
    print(f"Aiming for base samples per (Year, Topic) group: {base_samples_per_group}")
    print(f"Remainder samples to distribute: {remainder}")

    # 3. Perform initial sampling (base samples per group)
    np.random.seed(RANDOM_SEED) # Set seed for reproducibility

    # <<< MODIFIED LOOP: Iterate through (year, topic), group_df pairs >>>
    for (year, topic), group_df in grouped:
        group_indices = group_df.index.tolist()
        n_group = len(group_indices)

        # Determine how many to sample from this group initially
        n_to_sample = min(n_group, base_samples_per_group)

        # Sample indices
        if n_to_sample > 0: # Only sample if we need at least 1
            chosen_indices = np.random.choice(group_indices, size=n_to_sample, replace=False).tolist()
            sampled_indices.extend(chosen_indices)
        else:
            chosen_indices = [] # No samples taken from this group initially

        # Store remaining indices if the group was larger than base_samples_per_group
        if n_group > n_to_sample: # Check against actual sampled, not just base
             remaining_in_group = list(set(group_indices) - set(chosen_indices))
             if remaining_in_group: # Only store if there are actually remaining indices
                 # Use the tuple (year, topic) as the key
                 potential_extra_indices[(year, topic)] = remaining_in_group

    # 4. Distribute the remainder samples
    num_currently_sampled = len(sampled_indices)
    num_needed = TARGET_SAMPLES - num_currently_sampled # How many more we need

    print(f"Collected {num_currently_sampled} samples in the initial pass.")
    print(f"Need {num_needed} more samples for the remainder.")

    if num_needed > 0 and potential_extra_indices:
        # Create a flat list of all indices available for extra sampling
        # Each item in the list is an index from the original eligible_topic_df
        flat_potential_extras = [idx for indices_list in potential_extra_indices.values() for idx in indices_list]

        if flat_potential_extras:
             # Determine how many extras we can actually take
            n_extras_to_take = min(num_needed, len(flat_potential_extras))
            print(f"Sampling {n_extras_to_take} additional 'remainder' samples from the pool of {len(flat_potential_extras)} available extras...")

            # Sample from the pool of potential extras
            extra_indices_chosen = np.random.choice(flat_potential_extras, size=n_extras_to_take, replace=False).tolist()
            sampled_indices.extend(extra_indices_chosen)
        else:
             print("No potential extra indices found to draw remainder samples from, though some were needed.")

    elif num_needed > 0:
        print(f"Warning: Needed {num_needed} more samples for the remainder, but no groups had surplus rows after initial sampling.")


    # 5. Retrieve the final sampled DataFrame using the collected indices
    # <<< Make sure to sample from eligible_topic_df >>>
    final_samples_df = eligible_topic_df.loc[sampled_indices].copy()

    # Optional: Shuffle the final result so it's not ordered by year/topic groups
    final_samples_df = final_samples_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# --- Output ---
print(f"\n--- Sampling Complete ---")
print(f"Target number of samples: {TARGET_SAMPLES}")
print(f"Actual number of samples obtained: {len(final_samples_df)}")

if not final_samples_df.empty:
    print("\nDistribution of final samples by Year:")
    print(final_samples_df[YEAR_COL].value_counts().sort_index())

    print(f"\nDistribution of final samples by Top {n_topics_to_keep} Topics:")
    print(final_samples_df[TOPIC_COL].value_counts())

    # Verify distribution across Year and Topic combined (optional detail)
    # print("\nDistribution by Year and Topic combination:")
    # print(final_samples_df.groupby([YEAR_COL, TOPIC_COL]).size().unstack(fill_value=0))

    # print("\nFinal Sampled DataFrame (first 5 rows):")
    # print(final_samples_df.head())
    print("\nFinal Sampled DataFrame Info:")
    final_samples_df.info()

    try:
        final_samples_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully saved {len(final_samples_df)} samples to '{OUTPUT_CSV}'")
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")

else:
    print("\nNo samples were generated.")

print(f"Sum of 'Number of Refs': {sum(final_samples_df['Number of Refs'])}")
print(f"Shape of final df: {final_samples_df.shape}")
# Now 'final_samples_df' contains your desired sample set, filtered and stratified appropriately.