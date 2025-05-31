import pandas as pd
import re

# Load the data
df_check = pd.read_csv("abstract_check.csv")

# Filter for Suitability Score > 3
df_filtered = df_check[df_check["Suitability Score"] > 3].copy()

# Clean the 'Date (Year)' column: extract the first 4-digit number
df_filtered["Cleaned Year"] = df_filtered["Date (Year)"].astype(str).str.extract(r'(\d{4})')

# Now check the distribution of the cleaned year
year_distribution = df_filtered["Cleaned Year"].value_counts().sort_index()
print("Distribution of Cleaned Year:\n", year_distribution)

# Optionally, check topic distribution too
topic_distribution = df_filtered["Topic"].value_counts()
print("\nDistribution of Topics:\n", topic_distribution)
df_filtered.shape
df_filtered.to_csv("abstract_check_filtered_45.csv",index=False)