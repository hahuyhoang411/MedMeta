import pandas as pd
df = pd.read_csv("meta_analysis_summary.csv")

def clean_categories(df, drop_all_na=False):
    """
    Clean the Category column in the dataframe and remove rows with NA values.
    Categories are removed if they contain any of the terms in categories_to_remove (case insensitive).
    
    Args:
        df: pandas DataFrame with a 'Category' column
        drop_all_na: If True, drop rows with NA values in any column. 
                     If False, only drop rows with NA in the 'Category' column.
    
    Returns:
        DataFrame with cleaned 'Category' column and no NA values as specified
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Terms that should cause a category to be removed if they appear anywhere in the text
    categories_to_remove = [
            'report', 'short', 'article', 'review', 'research', 
        'original', 'online', 'topic', 'letter', 'images', 
        'studies', '临床研究', 'section', 'meta-analysis',
        'meta analysis', 'meta- analysis', 'meta-analyses',
        'data','meta‐analysis', 'meta‐analysis',
    ]
    
    # Special case conversions
    special_conversions = {
        'EATING DISORDERS: Edited by Hans W. Hoek': 'Eating disorders',
        'THROMBOSIS': 'Thrombosis'
    }
    
    # Function to clean a single category value
    def clean_category(category):
        # Handle NaN values
        if pd.isna(category):
            return None
        
        # Convert to string and strip whitespace
        category = str(category).strip()
        
        # Empty string check
        if category == '' or category.lower() == 'nan':
            return None
        
        # Check if this category contains any of the terms to remove (case insensitive)
        if any(remove_term.lower() in category.lower() for remove_term in categories_to_remove):
            return None
        
        # Check for special conversions
        if category in special_conversions:
            return special_conversions[category]
        
        # Check if category has a number
        if any(char.isdigit() for char in category):
            return None
            
        # Check if it's a category with an editor format
        if ':' in category and ('edited by' in category.lower() or 'editor' in category.lower()):
            # Extract the main category before the colon
            main_category = category.split(':')[0].strip()
            return main_category.capitalize()
            
        return category
    
    # Apply the cleaning function
    df['Category'] = df['Category'].apply(clean_category)
    
    # Drop rows with None/NA categories
    df = df.dropna(subset=['Category'])
    
    # Optionally drop rows with NA values in any column
    if drop_all_na:
        df = df.dropna()
    
    return df

cleaned_df = clean_categories(df, drop_all_na=False)
# Check the shape of your dataframes
print(f"Original dataframe shape: {df.shape}")
print(f"Cleaned dataframe shape (NA in Category dropped): {cleaned_df.shape}")
cleaned_df.to_csv("meta_analysis_summary_cleaned_topic.csv",index=False)