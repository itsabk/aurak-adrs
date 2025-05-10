# ADRS – Automated Dataset Recommendation System

> A research project by American University of Ras Al Khaimah, Computer Science Department

ADRS leverages Large Language Models (LLMs) via OpenRouter to help users discover relevant datasets on the Hugging Face Hub based on their natural language research intent.

It aims to demonstrate an alternative, potentially faster or more intuitive, approach compared to traditional keyword-based searching on the Hub.

## Features

- **Natural Language Input:** Describe the dataset you need in plain English.
- **LLM-Powered Keyword Generation:** Uses an LLM (configurable via OpenRouter) to extract relevant search keywords from your intent.
- **Advanced Search Options:** Specify dataset preferences like domain, task type, data size, license, quality criteria, languages, and time range.
- **Smart Filtering System:** Balances between strict filters and preference-based ranking to ensure adequate search results.
- **Hugging Face Hub Search:** Searches the Hugging Face Hub using the generated keywords.
- **LLM-Powered Evaluation & Ranking:** Concurrently evaluates datasets using the LLM based on dynamic criteria, standardizes metadata, assigns relevance scores, and ranks results.
- **LLM-Powered Reporting:** Generates a textual report synthesizing the discovery process, key findings, and data landscape analysis.
- **Integrated Visualizations:** Creates interactive charts (e.g., score distribution, domain counts, task types, licenses, size estimates, and relevance vs. popularity) to accompany the report.
- **Two Operating Modes:** Choose between "Auto" mode for fully automated workflow or "Assistive" mode for step-by-step confirmation.
- **Web Interface:** Simple UI built with Streamlit for easy interaction.
- **Configurable:** Set API keys, select the LLM model, add researcher profile, adjust search limits, and edit LLM prompts via the UI.
- **Fully Editable Prompts:** Edit the prompts for Keyword Generation, Criteria Generation, Dataset Evaluation, and Report Generation directly in the UI sidebar (Advanced).

## Setup

1.  **Clone the repository (or ensure you have the project files).**

2.  **Create and activate a Python virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    - Create a `.env` file in the project root (you can copy `.env.example`).
    - Add your OpenRouter API key:
      ```dotenv
      OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY_HERE
      ```
    - Optionally, add your Hugging Face Hub token (increases rate limits):
      ```dotenv
      HUGGINGFACE_HUB_TOKEN=YOUR_HUGGINGFACE_HUB_TOKEN_HERE
      ```
    - Alternatively, you can enter the keys directly into the Streamlit UI sidebar when running the app.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Open the URL** provided in your terminal (usually `http://localhost:8501`) in your web browser.

3.  **Configure (Sidebar):**

    - Ensure your OpenRouter API Key is entered (loaded from `.env` or pasted).
    - Optionally enter your Hugging Face Hub Token.
    - Optionally change the LLM Model identifier.
    - Optionally add your Researcher Profile to personalize recommendations.
    - Select operating mode ("Auto" or "Assistive").
    - Optionally configure Search Limits (`Max Keywords`, `Fetch Limit`, `Final Limit`).
    - Optionally edit the prompts used for all LLM interactions via the "📝 Edit Prompts" expander in the sidebar (Advanced).

4.  **Enter Intent:** Type your research intent into the main text area (e.g., "Datasets for sentiment analysis of product reviews") or click on an example to auto-fill.

5.  **Advanced Search Options (Optional):** Click the "🔍 Advanced Search Options" expander to specify preferences:

    - **Preference Filters** (green): These inform the LLM of your preferences for ranking but don't exclude datasets.
      - Domain: Select one or more domains or add custom domains.
      - Task Type: Select specific machine learning tasks or add custom tasks.
      - License: Specify license preferences or add custom licenses.
      - Quality Criteria: Select desired quality attributes or add custom criteria.
      - Languages: Specify language preferences or add custom languages.
    - **Strict Filters** (orange): These exclude datasets that don't match the criteria.
      - Data Size: Filter datasets by size category.
      - Time Range: Filter datasets by creation/update time range.

6.  **Click "Discover Datasets".**

7.  **Workflow Steps:**

    **Auto Mode:**

    - The system automatically generates keywords, criteria, searches Hugging Face Hub, evaluates datasets, and displays results.
    - Report and visualizations are automatically generated.

    **Assistive Mode:**

    - **Step 1:** Review and edit generated keywords
    - **Step 2:** Review and edit evaluation criteria
    - **Step 3:** System searches Hugging Face Hub and evaluates datasets
    - **Step 4:** System displays results
    - **Step 5:** Optionally generate report and visualizations

8.  **View Results:** The app displays a table of evaluated datasets from Hugging Face Hub, sorted by the LLM's relevance score.

9.  **View Report:** The system generates:
    - Interactive visualizations (score distribution, domain counts, task types, license distribution, etc.)
    - A detailed textual report generated by the LLM, summarizing the process and findings, and referencing the visualizations.

## How it Works

1.  **Input:** The user provides a research intent and optional advanced search preferences.

2.  **Mode Selection:** The user selects between "Auto" or "Assistive" mode.

3.  **Keyword Generation:** The intent and advanced search preferences are sent to the configured LLM (LiteLLM/OpenRouter) using the keyword generation prompt to generate relevant comma-separated search keywords.

4.  **Keyword Refinement (Assistive Mode):** The user confirms or edits the keywords.

5.  **Criteria Generation:** The user intent and advanced search preferences are sent to the LLM to generate a dynamic, task-specific list of evaluation criteria as a JSON list.

6.  **Criteria Refinement (Assistive Mode):** The user confirms or edits the evaluation criteria.

7.  **HF Search:** The keywords are used to search the Hugging Face Hub (`list_datasets` with `full=True`), respecting the configured search limits.

8.  **Aggregation:** Unique raw dataset metadata records are collected.

9.  **Filtering:**

    - Strict filters (data size, time range) are applied to datasets.
    - If too few datasets (< 5) match strict filters, the search is automatically expanded.
    - Preference filters are passed to the LLM for consideration during evaluation.

10. **Concurrent LLM Evaluation:**

    - The user intent, advanced search preferences, dynamic criteria, and raw metadata for each dataset are sent concurrently to the LLM.
    - The LLM evaluates each dataset against the criteria and assigns a relevance score (0.0-1.0).
    - The LLM extracts/infers standardized fields and provides reasoning for the score.

11. **Sorting & Display:** Datasets are sorted by relevance score and displayed in a table.

12. **Report Generation:**
    - In Auto mode: Automatically generates visualizations and a report.
    - In Assistive mode: User can click "Generate Full Report & Visualizations" button.
    - The system generates interactive visualizations including:
      - Relevance score distribution
      - Domain counts
      - Task type counts
      - License distribution
      - Data size estimates
      - Relevance score vs. popularity (downloads) scatter plot
    - The LLM generates a comprehensive report analyzing the datasets, considering advanced search preferences, and referencing the visualizations.

## System Components

The ADRS system consists of several key components:

1. **Web Interface (app.py):** Streamlit-based user interface for interaction and visualization.

2. **LLM Interaction (src/llm_interaction.py):** Handles communication with LLMs via LiteLLM/OpenRouter.

3. **Hugging Face Search (src/hf_search.py):** Manages dataset discovery on the Hugging Face Hub.

4. **Metadata Schema (src/metadata_schema.py):** Defines the structure for dataset metadata and evaluations.

5. **Advanced Search Schema (src/advanced_search_schema.py):** Defines structures and options for advanced search capabilities.

6. **Prompt Templates (prompts/):**
   - `generate_keywords.txt`: Extracts search keywords from user intent and advanced search preferences
   - `generate_criteria.txt`: Creates evaluation criteria based on user intent and advanced search preferences
   - `evaluate_dataset.txt`: Evaluates datasets against criteria
   - `generate_report.txt`: Synthesizes findings into a comprehensive report considering advanced search preferences

## Advanced Search Implementation

The advanced search system balances between strict filtering and preference-based ranking:

1. **Preference Filters (Soft Filters):**

   - Domain, Task Type, License, Quality Criteria, and Languages are used as context for the LLM.
   - These preferences guide the LLM during evaluation and ranking but do not exclude datasets.
   - The LLM considers how well each dataset aligns with these preferences when assigning scores.

2. **Strict Filters:**

   - Data Size and Time Range are applied as strict filters that can exclude datasets.
   - If fewer than 5 datasets match strict filters, the search is automatically expanded.
   - When search is expanded, the preferences are given higher priority during LLM evaluation.

3. **Custom Input:**

   - Each filter category allows adding custom values beyond predefined options.
   - Custom inputs are retained in session state and become available for future searches.
   - Custom inputs are passed to the LLM for consideration during evaluation.

4. **Visual Distinction:**
   - Preference filters have a green left border in the UI.
   - Strict filters have an orange left border in the UI.
   - Clear help text explains the difference between filter types.

This approach ensures the system finds relevant datasets while respecting important constraints, and leverages the LLM's capabilities to consider preferences during evaluation rather than through strict filtering.

## Attribution

ADRS is a research project developed by the Computer Science Department at American University of Ras Al Khaimah. This tool demonstrates the application of Large Language Models to improve dataset discovery workflows.

## Metadata Evaluation Format

After retrieving raw metadata from Hugging Face, the system uses an LLM to perform an integrated evaluation and standardization step based on the user's intent. The LLM analyzes the raw metadata and generates the following structure (defined in `src/metadata_schema.py` as `EvaluatedMetadata`). Fields are optional (`Optional[...]`) and will be included if the LLM can confidently extract, infer, or determine them.

```python
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
```

This integrated format provides both standardized information and an intent-based relevance assessment in a single step, with results sorted by the LLM's score.

## Future Enhancements

- Direct download of selected datasets
- More in-depth dataset analysis using LLMs
- Compatibility assessment for merging multiple datasets
- Custom filtering and sorting options
- Support for additional dataset repositories beyond Hugging Face
- Export options for reports and visualizations
- Enhanced collaborative research features
- Improved dataset usage suggestions based on researcher profiles
