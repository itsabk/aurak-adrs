# ADRS User Guide

This guide provides step-by-step instructions for using the Automated Dataset Recommendation System (ADRS).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key (get yours from https://openrouter.ai/keys)
- Hugging Face Hub token (optional, get yours from https://huggingface.co/settings/tokens)

### Installation

1. Clone the repository or ensure you have the project files
2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys by either:
   - Creating a `.env` file in the project root with your keys:
     ```
     OPENROUTER_API_KEY=your_openrouter_key_here
     HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
     ```
   - Or entering them directly in the app interface

### Launching the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will start and open in your default web browser, usually at http://localhost:8501.

## User Interface Overview

The ADRS interface consists of:

1. **Main Area**: Where you enter your research intent and view results
2. **Sidebar**: Configuration options for API keys, model selection, search limits, etc.
3. **Results Display**: Table of ranked datasets and visualization/report section

## Using ADRS

### Step 1: Configure the Application

1. Open the sidebar by clicking the ">" icon in the top-left corner
2. Enter your **OpenRouter API Key** (required)
3. Optionally enter your **Hugging Face Hub Token** (increases API rate limits)
4. Select a **LLM Model** or use the default
5. Choose the **Operating Mode**:
   - **Auto**: Fully automated workflow without user intervention
   - **Assistive**: Step-by-step workflow with user confirmation at key steps
6. Adjust **Search Limits** if needed:
   - **Max Keywords to Search**: Number of keywords to use (1-10)
   - **Fetch Limit per Keyword**: Datasets per keyword search (10-100)
   - **Final Results Limit**: Total datasets to return (10-200)
7. Close the sidebar by clicking anywhere in the main area or the "<" icon

### Step 2: Enter Your Research Intent

1. In the main text area, enter a description of the datasets you're looking for
   - Be specific about the domain, task, and any special requirements
   - Example: "Datasets for analyzing customer churn in the telecommunications sector"
2. Alternatively, click one of the example buttons below the text area to use a pre-defined intent

### Step 3: Start the Discovery Process

Click the **Discover Datasets** button to start the process.

### Step 4: Follow the Workflow

#### In Auto Mode:

The system will automatically:

1. Generate keywords
2. Generate evaluation criteria
3. Search Hugging Face Hub
4. Evaluate datasets
5. Display results
6. Generate visualizations and report

You'll see progress indicators as each step completes.

#### In Assistive Mode:

1. **Review Keywords**:

   - The system will generate keywords based on your intent
   - Review the keywords and edit them if needed
   - Click "Confirm Keywords and Continue"

2. **Review Evaluation Criteria**:

   - The system will generate evaluation criteria based on your intent
   - Review the criteria and edit them if needed
   - Click "Confirm Criteria and Search"

3. **Wait for Search and Evaluation**:

   - The system will search Hugging Face Hub and evaluate datasets
   - Progress indicators will show you the status

4. **Review Results**:
   - Explore the ranked datasets in the results table
   - Click "Generate Full Report & Visualizations" to create the report

### Step 5: Explore the Results

#### Ranked Datasets Table

The table displays datasets sorted by relevance score, including:

- Dataset ID and link to Hugging Face Hub
- Relevance score (0.0-1.0)
- LLM-generated reasoning for the score
- Standardized metadata (summary, domain, task type, etc.)

You can:

- Sort the table by clicking column headers
- Download the results as CSV by clicking "Download Results as CSV"
- View evaluation issues by expanding the "View Evaluation Issues" section

#### Synthesis Report

The report section includes:

- Interactive visualizations:
  - Relevance score distribution
  - Domain counts
  - Task type counts
  - License distribution
  - Data size estimates
  - Relevance score vs. popularity (downloads) scatter plot
- A comprehensive text report analyzing the datasets and referencing the visualizations

## Advanced Usage

### Editing Prompt Templates

Advanced users can customize the prompts used for LLM interactions:

1. Open the sidebar
2. Expand the "📝 Edit Prompts (Advanced)" section
3. Edit any of the prompt templates:
   - **Keyword Generation Prompt**: Used to extract keywords from user intent
   - **Criteria Generation Prompt**: Used to generate evaluation criteria
   - **Dataset Evaluation Prompt**: Used to evaluate datasets against criteria
   - **Report Generation Prompt**: Used to generate the synthesis report
4. Click "Reset All Prompts to Default" to revert to the original prompts

### Saving Results

To save your results:

- Use the "Download Results as CSV" button to save the dataset table
- Use your browser's print function (Ctrl+P or Cmd+P) and select "Save as PDF" to save the entire page including visualizations and report

## Troubleshooting

### Common Issues

1. **"OpenRouter API Key is missing"**:

   - Ensure you've entered a valid OpenRouter API key in the sidebar or `.env` file

2. **"An error occurred during keyword generation"**:

   - Check your internet connection
   - Verify your OpenRouter API key is valid
   - Try using a different LLM model

3. **"Could not parse LLM output"**:

   - The LLM response format was unexpected
   - System will use fallback approaches when possible
   - Try adjusting your intent to be more specific

4. **"No datasets were found"**:

   - Try different keywords or a broader intent
   - Increase the search limits in the sidebar
   - Consider using a Hugging Face Hub token for higher rate limits

5. **"Error evaluating dataset"**:
   - Individual dataset evaluation errors are displayed in the "View Evaluation Issues" section
   - These are usually due to rate limits or parsing issues
   - The system will continue with the datasets that were successfully evaluated

### Getting Help

If you encounter issues not covered here, please:

1. Check the terminal/console for error messages
2. Review the [Technical Documentation](technical_documentation.md) for more details
3. Open an issue on the project's GitHub repository
