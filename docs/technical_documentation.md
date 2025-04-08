# ADRS Technical Documentation

This document provides detailed technical information about the Automated Dataset Recommendation System (ADRS).

## System Architecture

ADRS is built on a modular architecture with the following key components:

1. **Web Interface (app.py)**: A Streamlit-based user interface that manages the application flow, user interactions, and visualizations.

2. **LLM Integration (src/llm_interaction.py)**: Handles communication with Large Language Models via LiteLLM and OpenRouter.

3. **Hugging Face Hub Search (src/hf_search.py)**: Manages dataset discovery on the Hugging Face Hub API.

4. **Metadata Schema (src/metadata_schema.py)**: Defines the standardized structure for dataset metadata and evaluation results.

5. **Prompt Templates (prompts/...)**: Text files containing prompt templates for different LLM interactions.

## Operating Modes

ADRS offers two distinct operating modes to accommodate different user preferences:

### Auto Mode

In Auto mode, the system executes all steps automatically without user intervention:

1. User enters their research intent
2. System generates keywords using LLM
3. System generates evaluation criteria using LLM
4. System searches Hugging Face Hub for datasets using keywords
5. System evaluates datasets using LLM
6. System displays sorted results
7. System automatically generates visualizations and report

### Assistive Mode

In Assistive mode, the system requires user confirmation at key steps:

1. User enters their research intent
2. System generates keywords using LLM
3. **User reviews and confirms/edits keywords**
4. System generates evaluation criteria using LLM
5. **User reviews and confirms/edits criteria**
6. System searches Hugging Face Hub for datasets using confirmed keywords
7. System evaluates datasets using LLM
8. System displays sorted results
9. **User can optionally generate visualizations and report**

## Workflow Details

### 1. User Intent Processing

The workflow begins with the user providing a research intent in natural language. This intent is used as the foundation for all subsequent steps.

### 2. Keyword Generation

The system uses the LLM to extract relevant search keywords from the user's intent:

- **Prompt Template**: `prompts/generate_keywords.txt`
- **Input**: User's research intent
- **Output**: Comma-separated list of relevant keywords
- **Implementation**: `get_llm_response()` in `src/llm_interaction.py`

### 3. Evaluation Criteria Generation

The system uses the LLM to generate evaluation criteria based on the user's intent:

- **Prompt Template**: `prompts/generate_criteria.txt`
- **Input**: User's research intent
- **Output**: JSON list of evaluation criteria
- **Implementation**: `get_llm_response()` in `src/llm_interaction.py`

### 4. Hugging Face Hub Search

The system searches the Hugging Face Hub using the generated/confirmed keywords:

- **Function**: `search_datasets()` in `src/hf_search.py`
- **Parameters**:
  - `keywords`: List of search keywords
  - `hf_token`: Optional Hugging Face Hub token
  - `max_keywords_to_search`: Maximum number of keywords to use (default: 5)
  - `fetch_limit_per_keyword`: Maximum datasets per keyword search (default: 50)
  - `final_result_limit`: Maximum total datasets to return (default: 100)
- **Output**: List of raw dataset metadata from Hugging Face Hub

### 5. Dataset Evaluation

The system evaluates each dataset using the LLM:

- **Prompt Template**: `prompts/evaluate_dataset.txt`
- **Function**: `evaluate_dataset_with_llm()` in `src/llm_interaction.py`
- **Inputs**:
  - `raw_metadata`: Raw dataset metadata from Hugging Face Hub
  - `user_intent`: User's research intent
  - `dynamic_criteria`: Generated/confirmed evaluation criteria
  - `evaluation_prompt_template`: Template for evaluation prompt
  - `model`: LLM model to use
- **Output**: Evaluated metadata with standardized fields and relevance score
- **Concurrent Execution**: Uses `ThreadPoolExecutor` for parallel processing

### 6. Results Display

The system displays the evaluated datasets in a sortable table with:

- Dataset ID and link to Hugging Face Hub
- Relevance score (0.0-1.0)
- LLM-generated reasoning for the score
- Standardized metadata fields (summary, domain, task type, etc.)

### 7. Report Generation

The system generates visualizations and a comprehensive report:

- **Visualizations**:
  - Relevance score distribution
  - Domain counts
  - Task type counts
  - License distribution
  - Data size estimates
  - Relevance score vs. popularity (downloads) scatter plot
- **Report Prompt Template**: `prompts/generate_report.txt`
- **Inputs**:
  - User intent
  - Keywords used
  - Evaluation criteria
  - Summary of evaluated datasets
- **Output**: Markdown-formatted report analyzing the datasets and referencing the visualizations

## Configuration Options

The system provides several configuration options through the Streamlit sidebar:

1. **API Keys**:

   - OpenRouter API Key (required)
   - Hugging Face Hub Token (optional, increases rate limits)

2. **LLM Model**: The OpenRouter model identifier to use for LLM interactions

3. **Operating Mode**: "Auto" or "Assistive"

4. **Search Limits**:

   - Max Keywords to Search (1-10)
   - Fetch Limit per Keyword (10-100)
   - Final Results Limit (10-200)

5. **Prompt Templates**: Editable templates for all LLM interactions

## Metadata Schema

The system uses a standardized metadata schema defined in `src/metadata_schema.py`:

```python
class EvaluatedMetadata(TypedDict, total=False):
    # --- Added by the system after LLM evaluation ---
    id: Optional[str]                  # Original Hugging Face Dataset ID
    url: Optional[str]                 # Direct URL to the Hugging Face dataset page
    downloads: Optional[int]           # Raw downloads count from HF
    likes: Optional[int]               # Raw likes count from HF

    # --- Extracted/Inferred/Evaluated by LLM ---
    clear_summary: Optional[str]       # Concise summary of the dataset's purpose and content
    domain: Optional[str]              # Primary domain (e.g., Healthcare, NLP)
    task_type: Optional[str]           # Primary ML task suitability (e.g., Classification)
    data_size_estimate: Optional[str]  # Estimated size (e.g., "Medium (1k-100k rows)")
    key_features_columns: Optional[List[str]] # List of key features, columns, or data fields
    data_quality_hints: Optional[List[str]] # Hints about data quality, collection, limitations
    potential_biases_mentioned: Optional[List[str]] # Potential biases explicitly mentioned
    license_type: Optional[str]        # Dataset license (e.g., 'apache-2.0', 'mit')
    relevance_score: Optional[float]   # LLM-assigned score (0.0-1.0) based on intent
    reasoning: Optional[str]           # LLM-generated explanation for the score
```

## Prompt Templates

The system uses several prompt templates stored in the `prompts/` directory:

1. **`generate_keywords.txt`**: Extracts search keywords from user intent
2. **`generate_criteria.txt`**: Creates evaluation criteria based on user intent
3. **`evaluate_dataset.txt`**: Evaluates datasets against criteria and standardizes metadata
4. **`generate_report.txt`**: Synthesizes findings into a comprehensive report
5. **`standardize_metadata.txt`**: Template for metadata standardization

## Error Handling

The system includes several error handling mechanisms:

1. **API Key Validation**: Checks for required API keys before starting
2. **LLM Response Parsing**: Robust parsing of LLM outputs with fallbacks
3. **Exception Handling**: Comprehensive try/except blocks with user-friendly error messages
4. **Evaluation Errors Tracking**: Separate list for tracking and displaying evaluation errors

## Performance Considerations

1. **Concurrent Evaluation**: Uses `ThreadPoolExecutor` for parallel dataset evaluation
2. **Progress Tracking**: Provides progress bars and status updates during evaluation
3. **Rate Limiting**: Respects Hugging Face Hub API rate limits
4. **Memory Management**: Efficient handling of dataset metadata

## Security Considerations

1. **API Key Storage**: API keys can be stored in `.env` file or entered directly in the UI
2. **No Data Persistence**: Does not store user data or API keys between sessions
3. **Input Validation**: Validates user inputs before processing

## Current Limitations

1. The system currently does not download or analyze the actual dataset contents, only metadata
2. Visualization options are predetermined and not customizable
3. No authentication or user management
4. No persistent storage of results
5. Limited to Hugging Face Hub datasets

## Future Enhancements

1. Direct download and analysis of selected datasets
2. Compatibility assessment for merging multiple datasets
3. Custom filtering and sorting options
4. Support for additional dataset repositories beyond Hugging Face
5. Export options for reports and visualizations
6. User authentication and result persistence
7. More advanced visualization options
