import pandas as pd
import numpy as np

# Load the dataframe
try:
    df = pd.read_csv("meta_analysis_summary.csv")
    print("DataFrame loaded successfully.")
    # Optional: Display first few rows and info to verify columns
    # print(df.head())
    # print(df.info())
except FileNotFoundError:
    print("Error: 'meta_analysis_summary.csv' not found. Make sure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- Parameters ---
TARGET_SAMPLES = 200
MIN_YEAR = 2018
MAX_YEAR = 2025
MAX_REFS = 20
YEAR_COL = 'Date (Year)' # Make sure this matches your exact column name
REFS_COL = 'Number of Refs' # Make sure this matches your exact column name
RANDOM_SEED = 42 # for reproducibility
OUTPUT_CSV = 'MedMeta.csv'
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

# Drop rows where conversion failed or values are missing in critical columns
df_cleaned = df.dropna(subset=[YEAR_COL, REFS_COL]).copy()
# Ensure year is integer after cleaning
df_cleaned[YEAR_COL] = df_cleaned[YEAR_COL].astype(int)

print(f"\nOriginal DataFrame shape: {df.shape}")
print(f"DataFrame shape after cleaning non-numeric/NaNs in '{YEAR_COL}'/'{REFS_COL}': {df_cleaned.shape}")


# 1. Filter the DataFrame based on the conditions
eligible_df = df_cleaned[
    (df_cleaned[YEAR_COL] >= MIN_YEAR) &
    (df_cleaned[YEAR_COL] <= MAX_YEAR) &
    (df_cleaned[REFS_COL] < MAX_REFS)
].copy()

print(f"\nNumber of rows eligible after filtering (Years {MIN_YEAR}-{MAX_YEAR}, Refs < {MAX_REFS}): {len(eligible_df)}")

# --- Sampling Logic ---
final_samples_df = pd.DataFrame() # Initialize an empty DataFrame

if len(eligible_df) == 0:
    print("No rows match the specified criteria. Cannot sample.")
elif len(eligible_df) <= TARGET_SAMPLES:
    print(f"Found {len(eligible_df)} eligible rows, which is less than or equal to the target {TARGET_SAMPLES}.")
    print("Returning all eligible rows.")
    final_samples_df = eligible_df.copy()
else:
    print(f"Attempting to sample {TARGET_SAMPLES} rows from {len(eligible_df)} eligible rows...")
    # 2. Stratified Sampling: Group by year
    grouped = eligible_df.groupby(YEAR_COL)
    n_groups = grouped.ngroups # Number of unique years found in the eligible data

    # Calculate base samples per year and the remainder
    base_samples_per_group = TARGET_SAMPLES // n_groups
    remainder = TARGET_SAMPLES % n_groups

    sampled_indices = []
    potential_extra_indices = {} # Keep track of remaining indices per group for remainder sampling

    print(f"Found {n_groups} unique years in the eligible data.")
    print(f"Base samples per year: {base_samples_per_group}")
    print(f"Remainder samples to distribute: {remainder}")

    # 3. Perform initial sampling (base samples per group)
    np.random.seed(RANDOM_SEED) # Set seed for reproducibility

    for year, group_df in grouped:
        group_indices = group_df.index.tolist()
        n_group = len(group_indices)

        # Determine how many to sample from this group initially
        n_to_sample = min(n_group, base_samples_per_group)

        # Sample indices
        chosen_indices = np.random.choice(group_indices, size=n_to_sample, replace=False).tolist()
        sampled_indices.extend(chosen_indices)

        # Store remaining indices if the group was larger than base_samples_per_group
        if n_group > base_samples_per_group:
             remaining_in_group = list(set(group_indices) - set(chosen_indices))
             if remaining_in_group: # Only store if there are actually remaining indices
                 potential_extra_indices[year] = remaining_in_group

    # 4. Distribute the remainder samples
    num_currently_sampled = len(sampled_indices)
    num_needed = TARGET_SAMPLES - num_currently_sampled # How many more we need (should roughly equal remainder)

    if num_needed > 0 and potential_extra_indices:
        # Create a flat list of all indices available for extra sampling
        flat_potential_extras = [idx for indices in potential_extra_indices.values() for idx in indices]

        if flat_potential_extras:
             # Determine how many extras we can actually take
            n_extras_to_take = min(num_needed, len(flat_potential_extras))
            print(f"Sampling {n_extras_to_take} additional 'remainder' samples...")

            # Sample from the pool of potential extras
            extra_indices_chosen = np.random.choice(flat_potential_extras, size=n_extras_to_take, replace=False).tolist()
            sampled_indices.extend(extra_indices_chosen)
        else:
             print("No potential extra indices found to draw remainder samples from, though some were needed.")

    elif num_needed > 0:
        print(f"Warning: Needed {num_needed} more samples for the remainder, but no groups had surplus rows.")


    # 5. Retrieve the final sampled DataFrame using the collected indices
    final_samples_df = eligible_df.loc[sampled_indices].copy()

    # Optional: Shuffle the final result so it's not ordered by year groups
    final_samples_df = final_samples_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# --- Output ---
print(f"\n--- Sampling Complete ---")
print(f"Target number of samples: {TARGET_SAMPLES}")
print(f"Actual number of samples obtained: {len(final_samples_df)}")

if not final_samples_df.empty:
    print("\nDistribution of samples by year:")
    print(final_samples_df[YEAR_COL].value_counts().sort_index())

    # print("\nFinal Sampled DataFrame (first 5 rows):")
    # print(final_samples_df.head())
    print("\nFinal Sampled DataFrame Info:")
    final_samples_df.info()
    final_samples_df.to_csv(OUTPUT_CSV,index=False)
else:
    print("\nNo samples were generated.")

# Now 'final_samples_df' contains your desired sample set.