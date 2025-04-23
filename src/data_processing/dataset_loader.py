import os
import logging
from typing import Dict, List, Optional

from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_abstract_and_language(batch):
    # Access the relevant lists from the batch
    abstract_texts = batch['MedlineCitation']['Article']['Abstract']['AbstractText']
    languages = batch['MedlineCitation']['Article']['Language']

    # Create a list of booleans: True if abstract is non-empty AND language is 'eng'
    keep_mask = [
        (text is not None and text != '') and (lang == 'eng')
        for text, lang in zip(abstract_texts, languages)
    ]
    return keep_mask

def transform_pubmed25_entry(example: Dict) -> Dict:
    """
    Transforms a single entry from the pubmed25 dataset to the desired format.

    Args:
        example: A dictionary representing a single row from the dataset.

    Returns:
        A dictionary with 'PMID', 'Year', and 'text' keys.
    """
    med_citation = example.get('MedlineCitation', {})
    article = med_citation.get('Article', {})

    pmid = med_citation.get('PMID', 'N/A') # Default to 'N/A' if missing
    # Safely get nested Year
    year = med_citation.get('DateRevised', {}).get('Year', 'N/A') # Default to 'N/A'

    title = article.get('ArticleTitle', '') # Default to empty string
    # Handle potential variations in Abstract structure
    abstract_data = article.get('Abstract', {})
    abstract_text = abstract_data.get('AbstractText', '') if isinstance(abstract_data, dict) else ''
    if isinstance(abstract_text, list): # Handle cases where AbstractText might be a list
        abstract_text = "\n".join(filter(None, abstract_text))


    # Construct the combined text field - Corrected "Tittle"
    text_content = f"# Title: {title}\n# Abstract: {abstract_text if abstract_text else 'No abstract found'}"

    return {
        'PMID': str(pmid) if pmid else 'N/A', # Ensure PMID is string
        'Year': str(year) if year else 'N/A', # Ensure Year is string
        'text': text_content
    }


def load_and_process_pubmed25(
    dataset_name: str = "HoangHa/pubmed25",
    split: str = "train",
    cache_dir: Optional[str] = "./cache",
    num_proc: Optional[int] = None
) -> Optional[Dataset]:
    """
    Loads the pubmed25 dataset, filters it, and transforms it.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        split: Dataset split to load (e.g., 'train').
        cache_dir: Directory for caching downloaded data.
        num_proc: Number of processes to use for map/filter. Defaults to os.cpu_count().

    Returns:
        The processed Hugging Face Dataset, or None if loading fails.
    """
    if num_proc is None:
        num_proc = os.cpu_count()
        logging.info(f"Using {num_proc} processes for dataset operations.")

    try:
        logging.info(f"Loading dataset '{dataset_name}' split '{split}'...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True, cache_dir=cache_dir)
        original_columns = dataset.column_names
        logging.info(f"Original dataset size: {len(dataset)}")

        logging.info("Filtering dataset for non-empty abstracts and English language...")
        filtered_dataset = dataset.filter(
            filter_abstract_and_language,
            batched=True, # Process in batches for efficiency
            num_proc=num_proc
        )
        logging.info(f"Filtered dataset size: {len(filtered_dataset)}")

        if len(filtered_dataset) == 0:
            logging.warning("Dataset is empty after filtering.")
            return None

        logging.info("Transforming dataset columns...")
        # Define expected features for the final dataset for robustness
        expected_features = Features({
            'PMID': Value(dtype='string', id=None),
            'Year': Value(dtype='string', id=None),
            'text': Value(dtype='string', id=None)
        })

        final_dataset = filtered_dataset.map(
            transform_pubmed25_entry,
            num_proc=num_proc,
            remove_columns=original_columns,
            features=expected_features # Apply the defined features
        )
        logging.info(f"Transformation complete. Final dataset size: {len(final_dataset)}")
        return final_dataset

    except Exception as e:
        logging.error(f"Failed to load or process dataset '{dataset_name}': {e}", exc_info=True)
        return None

def load_local_csv_dataset(file_path: str, split: str = 'train') -> Optional[Dataset]:
    """
    Loads a dataset from a local CSV file.

    Args:
        file_path: Path to the CSV file.
        split: Name to assign to the loaded split.

    Returns:
        The loaded Hugging Face Dataset, or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"CSV file not found: {file_path}")
        return None
    try:
        logging.info(f"Loading dataset from CSV: {file_path}")
        # Ensure columns are read as strings to avoid type issues later
        dataset = load_dataset('csv', data_files=file_path, split=split,
                                column_names=['PMID', 'Year', 'text'], # Define column names if no header
                                keep_in_memory=True) # Keep small ref dataset in memory
        logging.info(f"Successfully loaded CSV dataset with {len(dataset)} rows.")
        # Validate columns - datasets automatically handles this if column_names is set
        required_cols = {'PMID', 'Year', 'text'}
        if not required_cols.issubset(dataset.column_names):
            logging.error(f"CSV file {file_path} missing required columns. Found: {dataset.column_names}, Required: {required_cols}")
            return None
        return dataset
    except Exception as e:
        logging.error(f"Failed to load CSV dataset from '{file_path}': {e}", exc_info=True)
        return None

def concatenate_hf_datasets(datasets: List[Dataset]) -> Optional[Dataset]:
    """
    Concatenates a list of Hugging Face Datasets.

    Args:
        datasets: A list of datasets to concatenate. They must have compatible schemas.

    Returns:
        The concatenated dataset, or None if input is empty or concatenation fails.
    """
    if not datasets:
        logging.warning("Received empty list of datasets to concatenate.")
        return None
    # Optionally add schema validation here if needed
    try:
        logging.info(f"Concatenating {len(datasets)} datasets...")
        concatenated_dataset = concatenate_datasets(datasets)
        logging.info(f"Concatenated dataset size: {len(concatenated_dataset)}")
        return concatenated_dataset
    except Exception as e:
        logging.error(f"Failed to concatenate datasets: {e}", exc_info=True)
        return None