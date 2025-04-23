import logging
from typing import List, Iterator, Optional, Dict, Any
from datasets import Dataset
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomHuggingFaceDatasetLoader(BaseLoader):
    """
    Loads a Hugging Face Dataset into LangChain Documents.

    Each row in the dataset is treated as a separate Document.
    """

    def __init__(
        self,
        dataset: Dataset,
        page_content_column: str = "text",
        metadata_columns: Optional[List[str]] = None,
    ):
        """
        Initializes the loader.

        Args:
            dataset: The Hugging Face Dataset to load.
            page_content_column: The name of the column in the dataset
                                 to use as the Document's page_content.
                                 Defaults to "text".
            metadata_columns: A list of column names in the dataset to include
                              in the Document's metadata. If None, uses all columns
                              *except* the page_content_column.
                              Defaults to None.

        Raises:
            ValueError: If specified columns are not found in the dataset.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("Input 'dataset' must be a Hugging Face Dataset object.")

        self.dataset = dataset
        self.page_content_column = page_content_column

        dataset_cols = dataset.column_names
        if page_content_column not in dataset_cols:
             raise ValueError(
                 f"page_content_column '{page_content_column}' not found in dataset columns: {dataset_cols}"
             )

        if metadata_columns:
             # Validate provided metadata columns
             for col in metadata_columns:
                 if col not in dataset_cols:
                     raise ValueError(
                         f"metadata_column '{col}' not found in dataset columns: {dataset_cols}"
                     )
                 if col == page_content_column:
                      logging.warning(f"Metadata column '{col}' is the same as page_content_column. It will be included in metadata.")
             self.metadata_columns = metadata_columns
        else:
             # Default: use all other columns as metadata
             self.metadata_columns = [
                 col for col in dataset_cols if col != page_content_column
             ]
             logging.info(f"Using dataset columns for metadata: {self.metadata_columns}")


    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads documents from the Hugging Face Dataset.

        Yields:
            LangChain Document objects.
        """
        logging.info(f"Starting lazy load from Hugging Face Dataset ({len(self.dataset)} rows)...")
        for i, row in enumerate(self.dataset):
            try:
                # Extract page content
                page_content = row.get(self.page_content_column)
                # Ensure content is string, handle potential non-string data gracefully
                if not isinstance(page_content, str):
                    logging.debug(f"Row {i}: Converting non-string page content ({type(page_content)}) to string.")
                    page_content = str(page_content) if page_content is not None else ""

                # Extract metadata
                metadata: Dict[str, Any] = {}
                for col in self.metadata_columns:
                    metadata[col] = row.get(col) # Use .get for safety, handles missing keys

                yield Document(page_content=page_content, metadata=metadata)

            except Exception as e:
                 logging.error(f"Error processing row {i} from dataset: {e}. Row data: {row}", exc_info=True)
                 # Optionally skip the row or raise the exception
                 continue # Skip problematic row

        logging.info("Finished lazy loading documents.")

    def load(self) -> List[Document]:
        """Load all documents from the dataset into a list."""
        return list(self.lazy_load())