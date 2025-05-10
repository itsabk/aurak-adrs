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
5. Optionally add your **Researcher Profile** to personalize recommendations
6. Choose the **Operating Mode**:
   - **Auto**: Fully automated workflow without user intervention
   - **Assistive**: Step-by-step workflow with user confirmation at key steps
7. Adjust **Search Limits** if needed:
   - **Max Keywords to Search**: Number of keywords to use (1-10)
   - **Fetch Limit per Keyword**: Datasets per keyword search (10-100)
   - **Final Results Limit**: Total datasets to return (10-200)
8. Close the sidebar by clicking anywhere in the main area or the "<" icon

### Step 2: Enter Your Research Intent

1. In the main text area, enter a description of the datasets you're looking for
   - Be specific about the domain, task, and any special requirements
   - Example: "Datasets for analyzing customer churn in the telecommunications sector"
2. Alternatively, click one of the example buttons below the text area to auto-fill with a pre-defined intent

### Step 3: Specify Advanced Search Options (Optional)

1. Click the "🔍 Advanced Search Options" expander to open the advanced search panel
2. Set your preferences using the two types of filters:

   **Preference Filters** (green border):

   - These inform the LLM of your preferences but don't exclude datasets
   - Options include:
     - **Domain**: Select domains like NLP, Computer Vision, Finance, etc.
     - **Task Type**: Select ML tasks like Classification, Object Detection, etc.
     - **License**: Specify license preferences like MIT, Apache, etc.
     - **Quality Criteria**: Select quality requirements like Well-Documented, Clean, etc.
     - **Languages**: Specify language preferences like English, Spanish, etc.
   - For each, you can:
     - Select from predefined options
     - Add custom values using the "Add" button

   **Strict Filters** (orange border):

   - These will exclude datasets that don't match the criteria
   - Options include:
     - **Data Size**: Filter by size categories (Small, Medium, Large, etc.)
     - **Time Range**: Filter by creation/update time range
   - Note: If too few datasets (<5) match strict filters, the search will automatically expand

3. Use the "Reset Advanced Options" button to clear all filters if needed

### Step 4: Start the Discovery Process

Click the **Discover Datasets** button to start the process.

### Step 5: Follow the Workflow

#### In Auto Mode:

The system will automatically:

1. Generate keywords (incorporating advanced search preferences)
2. Generate evaluation criteria (incorporating advanced search preferences)
3. Search Hugging Face Hub
4. Apply strict filters (with automatic expansion if needed)
5. Evaluate datasets (considering preference filters)
6. Display results
7. Generate visualizations and report

You'll see progress indicators as each step completes.

#### In Assistive Mode:

1. **Review Keywords**:

   - The system will generate keywords based on your intent and preferences
   - Review the keywords and edit them if needed
   - Click "Confirm Keywords and Continue"

2. **Review Evaluation Criteria**:

   - The system will generate evaluation criteria based on your intent and preferences
   - Review the criteria and edit them if needed
   - Click "Confirm Criteria and Search"

3. **Wait for Search and Evaluation**:

   - The system will search Hugging Face Hub and evaluate datasets
   - You'll see messages if strict filters are applied or expanded
   - Progress indicators will show you the status

4. **Review Results**:
   - Explore the ranked datasets in the results table
   - Click "Generate Full Report & Visualizations" to create the report

### Step 6: Explore the Results

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

### Using Advanced Search Effectively

For best results when using advanced search:

1. **Start Broad**: Initially use fewer filters to ensure you get enough results
2. **Preference vs. Strict Filters**:
   - Use preference filters (green) for most requirements to inform ranking
   - Only use strict filters (orange) for critical requirements like size or recency
3. **Custom Values**: Add custom values when predefined options don't match your needs
4. **Combining Filters**: Multiple selected options within a filter are treated as OR conditions
5. **Filter Expansion**: If you see a message about "expanding search," it means too few datasets matched your strict filters

### Example Advanced Search Scenarios

**Scenario 1: Finding recent financial datasets**

- Domain (preference): Finance
- Time Range (strict): Last 2 years

**Scenario 2: Finding well-documented multilingual NLP datasets**

- Domain (preference): NLP
- Languages (preference): Select multiple languages
- Quality Criteria (preference): Well-Documented, Cleaned

**Scenario 3: Finding compact image datasets for mobile applications**

- Domain (preference): Computer Vision
- Task Type (preference): Image Classification
- Data Size (strict): Small (<1k samples)

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
   - Reduce the number of strict filters
   - Increase the search limits in the sidebar
   - Consider using a Hugging Face Hub token for higher rate limits

5. **"Error evaluating dataset"**:

   - Individual dataset evaluation errors are displayed in the "View Evaluation Issues" section
   - These are usually due to rate limits or parsing issues
   - The system will continue with the datasets that were successfully evaluated

6. **"Only X datasets match your strict filters. Expanding search..."**:
   - This is not an error but a message indicating the system automatically expanded your search
   - Your preferences will still be used for ranking
   - To get more focused results, try adjusting your research intent or using different preference filters

### Getting Help

If you encounter issues not covered here, please:

1. Check the terminal/console for error messages
2. Review the [Technical Documentation](technical_documentation.md) for more details
3. Open an issue on the project's GitHub repository
