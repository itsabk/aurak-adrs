from typing import TypedDict, List, Optional

# Original standardization schema (kept for reference or potential reuse if needed)
# class StandardizedMetadata(TypedDict):
#     """
#     Defines the standardized structure for dataset metadata after LLM processing.
#     """
#     clear_summary: Optional[str]
#     domain: Optional[str]
#     task_type: Optional[str]
#     data_size_estimate: Optional[str] # e.g., "Small (<1k rows)", "Medium (1k-100k rows)", "Large (>100k rows)"
#     key_features_columns: Optional[List[str]]
#     data_quality_hints: Optional[List[str]]
#     potential_biases_mentioned: Optional[List[str]]
#     license_type: Optional[str] 

# New schema for integrated LLM evaluation
class EvaluatedMetadata(TypedDict, total=False):
    """
    Defines the structure after integrated LLM evaluation, including both
    extracted/inferred fields and evaluation metrics based on user intent.
    Fields added *after* LLM call (like id, url, downloads, likes) are also included here.
    """
    # --- Added by the system after LLM evaluation --- 
    id: Optional[str]                 # Original Hugging Face Dataset ID
    url: Optional[str]                # Direct URL to the Hugging Face dataset page
    downloads: Optional[int]          # Raw downloads count from HF
    likes: Optional[int]              # Raw likes count from HF

    # --- Extracted/Inferred/Evaluated by LLM --- 
    clear_summary: Optional[str]        # Concise summary of the dataset's purpose and content.
    domain: Optional[str]               # Primary domain (e.g., Healthcare, NLP).
    task_type: Optional[str]            # Primary ML task suitability (e.g., Classification).
    data_size_estimate: Optional[str]   # Estimated size (e.g., "Medium (1k-100k rows)").
    key_features_columns: Optional[List[str]] # List of key features, columns, or data fields.
    data_quality_hints: Optional[List[str]] # Hints about data quality, collection, limitations.
    potential_biases_mentioned: Optional[List[str]] # Potential biases explicitly mentioned.
    license_type: Optional[str]         # Dataset license (e.g., 'apache-2.0', 'mit').
    relevance_score: Optional[float]    # LLM-assigned score (0.0-1.0) based on intent.
    reasoning: Optional[str]            # LLM-generated explanation for the score. 