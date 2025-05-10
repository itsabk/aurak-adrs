import streamlit as st
import pandas as pd
import json # Added json import
import litellm
from src.llm_interaction import get_llm_response, evaluate_dataset_with_llm, DEFAULT_MODEL
from src.hf_search import search_datasets
from src.metadata_schema import EvaluatedMetadata
import string
import html # Import the html module for escaping
import os
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm # For better progress tracking with concurrency
from typing import List
import traceback
import altair as alt # Import altair for potential advanced charts
import time # For potential UI updates/sleeps
import re # Import re for regular expressions
import streamlit.components.v1 as components
from streamlit_local_storage import LocalStorage # Import correct class

# --- Disable LiteLLM background logging to prevent thread pool conflicts ---
litellm.disable_streaming_logging = True

# Instantiate local storage
localS = LocalStorage()

# --- Default Prompt File Paths (used for initial load) ---
DEFAULT_CRITERIA_PROMPT_PATH = "prompts/generate_criteria.txt"
DEFAULT_EVALUATION_PROMPT_PATH = "prompts/evaluate_dataset.txt"
DEFAULT_KEYWORD_PROMPT_PATH = "prompts/generate_keywords.txt" # New default path
DEFAULT_REPORT_PROMPT_PATH = "prompts/generate_report.txt"   # New default path

# Load .env file for default keys (if available)
load_dotenv()

# --- Configuration Persistence Functions ---
# def load_config():
#     """Loads configuration from the JSON file."""
#     if os.path.exists(CONFIG_FILE):
#         try:
#             with open(CONFIG_FILE, 'r') as f:
#                 return json.load(f)
#         except (json.JSONDecodeError, IOError) as e:
#             st.warning(f"Could not load configuration file ({CONFIG_FILE}): {e}. Using defaults.")
#             return {}
#     return {}

# def save_config(config_dict):
#     """Saves the configuration dictionary to the JSON file."""
#     try:
#         with open(CONFIG_FILE, 'w') as f:
#             json.dump(config_dict, f, indent=4)
#     except IOError as e:
#         st.error(f"Could not save configuration to {CONFIG_FILE}: {e}")

# Load existing configuration - REMOVED
# loaded_config = load_config()

# --- Helper Functions ---
def proceed_with_search(keywords, user_intent, selected_model):
    """
    Function to continue with HF Hub search and dataset evaluation
    after keyword and criteria confirmation.
    """
    st.subheader("3. Searching Hugging Face Hub...")
    hf_token = st.session_state.huggingface_hub_token if st.session_state.huggingface_hub_token else None
    
    try:
        with st.spinner(f"Searching HF Hub based on top keywords: {keywords[:3]}..."):
            # Pass HF token to search function
            # Pass configured limits to search function
            st.session_state.datasets = search_datasets(
                keywords, 
                hf_token=hf_token,
                max_keywords_to_search=st.session_state.max_keywords,
                fetch_limit_per_keyword=st.session_state.fetch_limit,
                final_result_limit=st.session_state.final_limit
            )
            # Mark search as done right after it completes
            st.session_state.search_triggered = True 
            st.session_state.error_message = None # Clear previous errors if search succeeds
        
        # ---> Step 4: Integrated LLM Evaluation (using criteria) <--- #
        if st.session_state.datasets:
            st.subheader("4. Evaluating Datasets with LLM (using confirmed criteria)...")
            # Create a map of raw dataset ID to its full data for easy lookup
            raw_datasets_map = {d.get('id'): d for d in st.session_state.datasets if d.get('id')}
            
            llm_model_to_use = selected_model if selected_model else DEFAULT_MODEL
            total_datasets = len(st.session_state.datasets)
            status_text = st.empty()
            progress_bar = st.progress(0.0) # Initialize progress bar
            status_text.text(f"Starting evaluation for {total_datasets} datasets using confirmed criteria...")

            # Use ThreadPoolExecutor for concurrency
            MAX_WORKERS = 10 # Adjust based on API rate limits and desired concurrency
            processed_count = 0
            futures_to_dataset = {}
            
            with st.spinner(f"Sending {total_datasets} datasets to LLM for evaluation based on your intent (max {MAX_WORKERS} concurrent requests)..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Submit all tasks
                    for dataset_raw in st.session_state.datasets:
                        future = executor.submit(
                            evaluate_dataset_with_llm, # Use new function
                            raw_metadata=dataset_raw, 
                            user_intent=user_intent, # Pass user intent
                            dynamic_criteria=st.session_state.dynamic_criteria, # Pass confirmed criteria
                            # Pass prompt template content
                            evaluation_prompt_template=st.session_state.evaluation_prompt_content,
                            model=llm_model_to_use,
                            researcher_profile=st.session_state.get('researcher_profile', 'No profile provided')
                        )
                        futures_to_dataset[future] = dataset_raw # Map future back to original data

                    # Process results as they complete
                    for i, future in enumerate(concurrent.futures.as_completed(futures_to_dataset)):
                        dataset_raw = futures_to_dataset[future]
                        dataset_id = dataset_raw.get('id', 'N/A')
                        try:
                            llm_evaluation_result = future.result() 
                            
                            if llm_evaluation_result:
                                # Result contains LLM fields (summary, score, etc.)
                                # Now, add the essential non-LLM fields
                                evaluated_metadata = llm_evaluation_result.copy()
                                
                                dataset_id = dataset_raw.get('id', 'N/A') # Get ID from the *raw* data
                                evaluated_metadata['id'] = dataset_id
                                evaluated_metadata['url'] = f"https://huggingface.co/datasets/{dataset_id}"
                                
                                # Add raw metrics from original data
                                raw_data = raw_datasets_map.get(dataset_id)
                                if raw_data:
                                    evaluated_metadata['downloads'] = raw_data.get('downloads')
                                    evaluated_metadata['likes'] = raw_data.get('likes')
                                else:
                                    evaluated_metadata['downloads'] = 0
                                    evaluated_metadata['likes'] = 0
                                
                                # Append the completed record
                                st.session_state.evaluated_datasets.append(evaluated_metadata) 
                            else:
                                # LLM function returned None (e.g., parsing error)
                                err_msg = f"⚠️ Dataset {dataset_id}: LLM evaluation/parsing failed."
                                st.session_state.evaluation_errors.append(err_msg)
                                st.toast(err_msg, icon="⚠️")
                        except Exception as exc:
                            # Exception raised during the execution of the future OR processing the result
                            dataset_id = dataset_raw.get('id', 'N/A') # Need ID for message
                            err_msg = f"🔥 Dataset {dataset_id}: Error during evaluation - {type(exc).__name__}: {exc}"
                            st.session_state.evaluation_errors.append(err_msg)
                            # --- Add Detailed Exception Logging Here ---
                            print(f"--- ERROR CAUGHT IN APP.PY for dataset {dataset_id} ---")
                            print(f"Exception Type: {type(exc).__name__}")
                            print(f"Exception Args: {exc.args}")
                            print(f"Exception __str__: {exc}")
                            print(f"Exception __repr__: {repr(exc)}")
                            print("Traceback from app.py:")
                            traceback.print_exc() # Print traceback seen by this loop
                            print("--- END ERROR CAUGHT IN APP.PY ---")
                            # --- End Detailed Logging ---
                            st.toast(f"🔥 Error evaluating {dataset_id}: {exc}", icon="🔥")
                        
                        processed_count += 1
                        # Update progress bar
                        progress_percentage = processed_count / total_datasets
                        status_text.text(f"Evaluating dataset {processed_count}/{total_datasets} ({dataset_id})...")
                        progress_bar.progress(progress_percentage)
            
            # Ensure progress bar completes and status updates
            progress_bar.progress(1.0)
            status_text.text(f"Evaluation complete. Processed {len(st.session_state.evaluated_datasets)} / {total_datasets} datasets.")

            # --- Step 5: Sorting Step (based on LLM score) ---
            if st.session_state.evaluated_datasets:
                st.subheader("5. Sorting Datasets by Relevance Score...")
                with st.spinner("Sorting datasets based on LLM evaluation..."):
                    st.session_state.evaluated_datasets.sort(
                        key=lambda x: x.get('relevance_score', 0.0), 
                        reverse=True # Higher score first
                    )
                st.success(f"Sorting complete. Displaying {len(st.session_state.evaluated_datasets)} evaluated datasets.")
            else:
                st.info("No datasets were successfully evaluated, skipping sorting.")
            # --- End Sorting Step ---
    except Exception as e:
        st.session_state.error_message = f"An error occurred during Hugging Face search or processing: {e}"
        st.error(st.session_state.error_message)
        st.exception(e) # Show full traceback for errors
        st.session_state.search_triggered = False # Ensure search is marked as not done on error
        st.session_state.datasets = [] # Clear datasets on error
        st.session_state.evaluated_datasets = [] # Also clear evaluated datasets


st.set_page_config(
    layout="wide", 
    page_title="ADRS - Automated Dataset Recommendation System",
    page_icon="assets/adrs-icon.png",
    initial_sidebar_state="collapsed"
)

# Add auto-close sidebar JavaScript
sidebar_close_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Function to detect clicks outside sidebar and close it
    function setupSidebarAutoClose() {
        const mainArea = document.querySelector('section.main');
        const sidebar = document.querySelector('section.css-1cypcdb.e1akgbir11'); // Streamlit sidebar class
        
        if (mainArea && sidebar) {
            mainArea.addEventListener('click', function(e) {
                // Check if sidebar is expanded
                if (document.querySelector('.css-1cypcdb.e1akgbir11[aria-expanded="true"]')) {
                    // Emit click on the collapse button (hamburger menu)
                    const collapseButton = document.querySelector('button[kind="header"]');
                    if (collapseButton) {
                        collapseButton.click();
                    }
                }
            });
        }
    }
    
    // Give Streamlit time to load UI
    setTimeout(setupSidebarAutoClose, 1000);
    
    // Also run when Streamlit rerun happens 
    const observer = new MutationObserver(function(mutations) {
        setTimeout(setupSidebarAutoClose, 1000);
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""
components.html(sidebar_close_js, height=0, width=0)

# Display logo and title in a row
col1, col2 = st.columns([1, 4])
with col1:
    # Remove logo from hero
    pass
with col2:
    st.title("ADRS – Automated Dataset Recommendation System")

# --- Session State Initialization ---
# Fetch initial values from local storage
# local_storage_keys = ["openrouter_api_key", "huggingface_hub_token", "selected_model"]
# Need to provide default values for get_items if key might not exist
# defaults_for_get = {
#     "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
#     "huggingface_hub_token": os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
#     "selected_model": DEFAULT_MODEL
# }
# initial_config = get_items(local_storage_keys, default_value=defaults_for_get, key="get_init_config")

# Fetch individual items using LocalStorage().getItem
initial_or_key = localS.getItem('openrouter_api_key')
initial_hf_token = localS.getItem('huggingface_hub_token')
initial_model = localS.getItem('selected_model')
initial_profile = localS.getItem('researcher_profile')

# Initialize session state variables if they don't exist
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = initial_or_key if initial_or_key else os.getenv("OPENROUTER_API_KEY", "")
if 'huggingface_hub_token' not in st.session_state:
    st.session_state.huggingface_hub_token = initial_hf_token if initial_hf_token else os.getenv("HUGGINGFACE_HUB_TOKEN", "")
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = initial_model if initial_model else DEFAULT_MODEL
if 'researcher_profile' not in st.session_state:
    st.session_state.researcher_profile = initial_profile if initial_profile else ""

# Initialize keys in session state if they don't exist
# Use local storage values first, then env vars, then hardcoded
default_values = {
    'keywords_generated': False,
    'generated_keywords': [],
    'refined_keywords_str': "",
    'search_triggered': False,
    'datasets': [],
    'evaluated_datasets': [],
    'dynamic_criteria': None,
    'criteria_confirmed': False,  # Add new flag for tracking criteria confirmation
    'workflow_step': 0,  # Add workflow state tracking: 0=start, 1=keywords, 2=criteria, 3=search/results
    'final_keywords': [],  # Store final keywords to use after confirmation
    'error_message': None,
    # --- Added prompt content state ---
    'criteria_prompt_content': None,
    'evaluation_prompt_content': None,
    'keyword_prompt_content': None, # New state for keyword prompt
    'report_prompt_content': None,  # New state for report prompt
    # --- End added state ---
    'max_keywords': 5,
    'fetch_limit': 50,
    'final_limit': 100,
    'generated_report': None, # Add state for the generated report
    'report_charts': None, # Add state for generated charts
    'evaluation_errors': [], # Add state for evaluation errors
    'user_intent_input': None, # For handling example buttons (deprecated by direct key use)
    'mode': 'auto', # Default to auto mode
    # Add keys for the input widgets to manage state before saving
    'openrouter_api_key_input': initial_or_key if initial_or_key is not None else os.getenv("OPENROUTER_API_KEY", ""),
    'huggingface_hub_token_input': initial_hf_token if initial_hf_token is not None else os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
    'selected_model_input': initial_model if initial_model is not None else DEFAULT_MODEL,
    'user_intent_main_input': "", # Initialize main input area state
    'criteria_editing_mode': False,  # Track if we're in criteria editing mode
    'criteria_generated': False,     # Track if criteria were generated by LLM
    'current_criteria': None,        # Store the current editable criteria
}
for key, default_value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Function to load initial prompt content --- 
def load_prompt_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Initial prompt file not found: {filepath}. Please check the path or create the file.")
        return f"Error: File not found at {filepath}"
    except Exception as e:
        st.error(f"Error reading initial prompt file {filepath}: {e}")
        return f"Error reading file: {e}"

# Load initial prompt content into session state if it's None
if st.session_state.criteria_prompt_content is None:
    st.session_state.criteria_prompt_content = load_prompt_from_file(DEFAULT_CRITERIA_PROMPT_PATH)
if st.session_state.evaluation_prompt_content is None:
    st.session_state.evaluation_prompt_content = load_prompt_from_file(DEFAULT_EVALUATION_PROMPT_PATH)
# Load new prompts
if st.session_state.keyword_prompt_content is None:
    st.session_state.keyword_prompt_content = load_prompt_from_file(DEFAULT_KEYWORD_PROMPT_PATH)
if st.session_state.report_prompt_content is None:
    st.session_state.report_prompt_content = load_prompt_from_file(DEFAULT_REPORT_PROMPT_PATH)

# --- Callback function to save config ---
def _save_config_callback():
    """Callback to save configuration when sidebar inputs change."""
    config_to_save = {
        'openrouter_api_key_input': st.session_state.get('openrouter_api_key_input', ''),
        'huggingface_hub_token_input': st.session_state.get('huggingface_hub_token_input', ''),
        'selected_model_input': st.session_state.get('selected_model_input', DEFAULT_MODEL),
        'researcher_profile_input': st.session_state.get('researcher_profile_input', '')
    }
    # Update main session state keys from the input keys
    st.session_state.openrouter_api_key = config_to_save['openrouter_api_key_input']
    st.session_state.huggingface_hub_token = config_to_save['huggingface_hub_token_input']
    st.session_state.selected_model = config_to_save['selected_model_input']
    st.session_state.researcher_profile = config_to_save['researcher_profile_input']
    # Save to local storage instead of file
    # Provide unique keys for each setItem call within the callback
    localS.setItem("openrouter_api_key", config_to_save['openrouter_api_key_input'], key="ls_set_or_key")
    localS.setItem("huggingface_hub_token", config_to_save['huggingface_hub_token_input'], key="ls_set_hf_key")
    localS.setItem("selected_model", config_to_save['selected_model_input'], key="ls_set_model")
    localS.setItem("researcher_profile", config_to_save['researcher_profile_input'], key="ls_set_profile")
    # save_config(config_to_save) # REMOVED

# --- Sidebar for Configuration ---
# Add logo to sidebar (moved above the header)
st.sidebar.image("assets/adrs-logo.png", width=300)
st.sidebar.header("Configuration")
st.session_state.openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    help="Get yours from https://openrouter.ai/keys",
    key="openrouter_api_key_input", # Assign a key
    on_change=_save_config_callback # Add callback
)
st.session_state.huggingface_hub_token = st.sidebar.text_input(
    "Hugging Face Hub Token",
    type="password",
    help="Increases API rate limits. Get yours from https://huggingface.co/settings/tokens",
    key="huggingface_hub_token_input", # Assign a key
    on_change=_save_config_callback # Add callback
)
st.sidebar.markdown("--- ")
st.sidebar.header("Settings")
# Use the main state key for selected_model here, as callback updates it
selected_model = st.sidebar.text_input(
    "LLM Model (OpenRouter)",
    key="selected_model_input", # Assign a key
    on_change=_save_config_callback # Add callback
)

# Add researcher profile section
st.sidebar.markdown("--- ")
st.sidebar.header("Researcher Profile")
researcher_profile = st.sidebar.text_area(
    "Your Research Background",
    value=st.session_state.get('researcher_profile', ''),
    help="Optional: Describe your research background, interests, and domain expertise to help bias dataset recommendations.",
    key="researcher_profile_input",
    height=100,
    placeholder="e.g., I am a computer vision researcher specializing in medical imaging analysis, with experience in deep learning for tumor detection.",
    on_change=_save_config_callback
)

# --- Add Mode Toggle UI ---
st.sidebar.markdown("--- ")
st.sidebar.header("Mode")
mode_options = ["Assistive", "Auto"]
selected_mode = st.sidebar.radio(
    "Select Mode",
    mode_options,
    index=0 if st.session_state.mode == "assistive" else 1,
    help="Assistive: Asks for confirmation at each step\nAuto: Executes all steps automatically"
)
# Update session state based on selection
st.session_state.mode = selected_mode.lower()

# --- Add Editable Prompts UI --- 
with st.sidebar.expander("📝 Edit Prompts (Advanced)", expanded=False):
    st.caption("Modify the prompts used for LLM interaction in this session.")
    st.session_state.criteria_prompt_content = st.text_area(
        label="Criteria Generation Prompt",
        value=st.session_state.criteria_prompt_content,
        height=250,
        key="criteria_prompt_editor"
    )
    st.session_state.evaluation_prompt_content = st.text_area(
        label="Dataset Evaluation Prompt",
        value=st.session_state.evaluation_prompt_content,
        height=400,
        key="evaluation_prompt_editor"
    )
    # Add editors for new prompts
    st.session_state.keyword_prompt_content = st.text_area(
        label="Keyword Generation Prompt",
        value=st.session_state.keyword_prompt_content,
        height=150, # Adjust height as needed
        key="keyword_prompt_editor"
    )
    st.session_state.report_prompt_content = st.text_area(
        label="Report Generation Prompt",
        value=st.session_state.report_prompt_content,
        height=400, # Adjust height as needed
        key="report_prompt_editor"
    )
    # Add prompt reset button
    if st.button("🔁 Reset All Prompts to Default", key="reset_prompts"): 
        st.session_state.keyword_prompt_content = load_prompt_from_file(DEFAULT_KEYWORD_PROMPT_PATH)
        st.session_state.criteria_prompt_content = load_prompt_from_file(DEFAULT_CRITERIA_PROMPT_PATH)
        st.session_state.evaluation_prompt_content = load_prompt_from_file(DEFAULT_EVALUATION_PROMPT_PATH)
        st.session_state.report_prompt_content = load_prompt_from_file(DEFAULT_REPORT_PROMPT_PATH)
        # Use experimental rerun to update text areas immediately
        st.rerun()
# --- End Editable Prompts UI --- 

st.sidebar.markdown("###### Search Limits")
st.session_state.max_keywords = st.sidebar.number_input(
   "Max Keywords to Search", 
   min_value=1, 
   max_value=10, 
   value=st.session_state.max_keywords, 
   step=1,
   help="How many of the top generated keywords should be used for individual HF Hub searches."
)
st.session_state.fetch_limit = st.sidebar.number_input(
   "Fetch Limit per Keyword", 
   min_value=10, 
   max_value=100, 
   value=st.session_state.fetch_limit, 
   step=5,
   help="Maximum datasets to retrieve from HF Hub for each individual keyword search."
)
st.session_state.final_limit = st.sidebar.number_input(
   "Final Results Limit", 
   min_value=10, 
   max_value=200, 
   value=st.session_state.final_limit, 
   step=10,
   help="Maximum number of unique datasets to display after combining search results."
)

# --- Main Area ---
# st.write("Enter your research intent below, configure API keys/settings in the sidebar, then click Discover.")

# Add workflow step indicator
if st.session_state.mode == "assistive" and st.session_state.workflow_step > 0:
    steps = ["Intent", "Keywords", "Criteria", "Search & Results", "Report"]
    current_step = min(st.session_state.workflow_step + 1, 4)  # +1 because step 0 is Intent input
    
    # Create progress indicator container with light gray background
    progress_container = st.container()
    with progress_container:
        # Add a light gray background
        st.markdown("---")
        st.markdown("### Workflow Progress")
        
        # Create columns for each step
        step_cols = st.columns(len(steps))
        
        # Fill each column with the appropriate step information
        for i, (col, step) in enumerate(zip(step_cols, steps)):
            with col:
                if i < current_step:
                    # Completed step
                    st.success(step)
                elif i == current_step:
                    # Current step
                    st.error(step)
                else:
                    # Pending step
                    st.info(step)
        
        st.markdown("---")

user_intent = st.text_area("Enter your research intent:",
                         height=100,
                         placeholder="e.g., Datasets for analyzing customer churn in the telecommunications sector",
                         key="user_intent_main_input") # Key is sufficient

# Update session state when text area changes (necessary for example buttons to work seamlessly)
# No longer needed - st.session_state.user_intent_input = user_intent

# Display mode information before discovery button
st.info(f"Current Mode: **{st.session_state.mode.capitalize()}**  \n"
        f"{'✅ Auto mode will execute all steps automatically without confirmation' if st.session_state.mode == 'auto' else '✅ Assistive mode will ask for confirmation at each step'}")

# Add debugging expander
with st.expander("🔍 Debug Info", expanded=False):
    st.write("Current Workflow Step:", st.session_state.workflow_step)
    st.write("Keywords Generated:", st.session_state.keywords_generated)
    st.write("Keywords:", st.session_state.generated_keywords)
    st.write("Search Triggered:", st.session_state.search_triggered)
    st.write("Criteria Confirmed:", st.session_state.criteria_confirmed)
    if st.session_state.error_message:
        st.error(f"Error: {st.session_state.error_message}")

# --- Example Intents --- 
with st.expander("📋 Try an example intent", expanded=False):
    examples = [
        "Datasets for analyzing customer churn in the telecommunications sector",
        "Images of street scenes for autonomous driving object detection",
        "Financial time series data for stock market prediction",
        "Medical imaging datasets for tumor segmentation"
    ]

# Define callback function BEFORE the loop
def update_intent_input(example_text):
    st.session_state.user_intent_main_input = example_text

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    # Use on_click for the button
    cols[i].button(f"Example {i+1}", 
                   help=example, 
                   key=f"ex_btn_{i}",
                   on_click=update_intent_input, 
                   args=(example,)) 

st.divider() # Visual separation

# --- Keyword Generation ---
if st.session_state.workflow_step == 0 and st.button("Discover Datasets"):
    # Reset state for a new discovery
    st.session_state.keywords_generated = False
    st.session_state.generated_keywords = []
    st.session_state.refined_keywords_str = ""
    st.session_state.search_triggered = False
    st.session_state.datasets = []
    st.session_state.evaluated_datasets = []
    st.session_state.dynamic_criteria = None
    st.session_state.criteria_confirmed = False  # Reset criteria confirmation
    st.session_state.workflow_step = 0  # Reset workflow state
    st.session_state.final_keywords = []  # Reset stored keywords
    st.session_state.error_message = None
    st.session_state.generated_report = None # Reset report on new discovery
    st.session_state.report_charts = None # Reset charts on new discovery
    st.session_state.evaluation_errors = [] # Reset evaluation errors

    # Use the intent from the session state which reflects text area content
    user_intent = st.session_state.user_intent_main_input

    # --- Pre-flight Check ---
    if not st.session_state.openrouter_api_key:
        st.error("OpenRouter API Key is missing. Please configure it in the sidebar.")
    elif not user_intent:
        st.warning("Please enter your research intent.")
    else:
        # Set LiteLLM API Key dynamically
        litellm.api_key = st.session_state.openrouter_api_key

        try:
            st.subheader("1. Generating Keywords...")
            with st.spinner("Asking LLM to generate keywords..."):
                # Use prompt content from session state
                keyword_prompt_template = st.session_state.keyword_prompt_content
                if not keyword_prompt_template or keyword_prompt_template.startswith("Error:"):
                     raise ValueError("Keyword Generation Prompt is missing or invalid. Check sidebar settings or create prompts/generate_keywords.txt.")
                
                # Format the prompt (ensure it contains {user_intent})
                formatted_keyword_prompt = keyword_prompt_template.format(
                    user_intent=user_intent,
                    researcher_profile=st.session_state.get('researcher_profile', 'No profile provided')
                )
                
                llm_model_to_use = st.session_state.selected_model # Use model from session state
                raw_llm_response = get_llm_response(formatted_keyword_prompt, model=llm_model_to_use)

                lines = [line.strip() for line in raw_llm_response.strip().split('\n') if line.strip()]
                keyword_string = lines[-1] if lines else ""
                punctuation_to_remove = string.punctuation.replace("_", "")
                translator = str.maketrans('', '', punctuation_to_remove)
                keywords = [
                    kw.translate(translator).strip()
                    for kw in keyword_string.split(',')
                    if kw.translate(translator).strip()
                ]
                st.session_state.generated_keywords = [kw for kw in keywords if len(kw.split()) < 5]

            if st.session_state.generated_keywords:
                st.session_state.keywords_generated = True
                st.session_state.refined_keywords_str = ", ".join(st.session_state.generated_keywords)
                
                # Check mode and handle workflow accordingly
                if st.session_state.mode == "auto":
                    # In auto mode, automatically use generated keywords
                    st.success(f"Generated Keywords: `{', '.join(st.session_state.generated_keywords)}`")
                    st.session_state.final_keywords = st.session_state.generated_keywords
                    st.session_state.workflow_step = 2  # Skip to criteria generation step
                    
                    # Automatically generate criteria
                    st.subheader("2. Generating Evaluation Criteria...")
                    with st.spinner("Asking LLM to generate evaluation criteria based on your intent..."):
                        try:
                            # Use prompt content from session state
                            criteria_prompt_template = st.session_state.criteria_prompt_content
                            if not criteria_prompt_template or criteria_prompt_template.startswith("Error:"):
                                raise ValueError("Criteria Generation Prompt is missing or invalid. Check sidebar settings.")
                            
                            formatted_criteria_prompt = criteria_prompt_template.format(user_intent=user_intent)
                            criteria_llm_output = get_llm_response(prompt=formatted_criteria_prompt, model=st.session_state.selected_model)
                            
                            # Parse the criteria (expected JSON list of strings)
                            try:
                                # --- Improved Parsing Logic --- 
                                json_str_crit = criteria_llm_output.strip()
                                if json_str_crit.startswith("```json"):
                                    json_str_crit = json_str_crit[7:].strip() # Remove ```json
                                if json_str_crit.endswith("```"):
                                    json_str_crit = json_str_crit[:-3].strip() # Remove ```
                                
                                # Final strip just in case
                                json_str_crit = json_str_crit.strip()
                                
                                parsed_criteria = json.loads(json_str_crit)
                                # --- End Improved Parsing Logic ---
                                
                                if isinstance(parsed_criteria, list) and all(isinstance(item, str) for item in parsed_criteria):
                                    st.session_state.dynamic_criteria = parsed_criteria
                                    st.success("Successfully generated evaluation criteria.")
                                else:
                                    raise ValueError("LLM output for criteria was not a list of strings.")
                            except (json.JSONDecodeError, ValueError) as crit_e:
                                st.warning(f"Could not parse LLM output for criteria as a valid JSON list of strings: {crit_e}")
                                st.text_area("LLM Criteria Output (raw):", criteria_llm_output, height=100, disabled=True)
                                st.info("Proceeding with default criteria. Evaluation might be less specific.")
                                # Ensure fallback is also stored (as a list)
                                st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
                            
                        except ValueError as ve:
                            # Raised if prompt content is bad
                            st.error(f"Prompt Error: {ve}")
                            raise # Stop execution if prompt is missing
                        except Exception as gen_crit_e:
                            st.error(f"An error occurred during criteria generation: {gen_crit_e}")
                            # Decide if we should proceed with a fallback or stop
                            st.info("Proceeding with default criteria due to error. Evaluation might be less specific.")
                            # Ensure fallback is also stored (as a list)
                            st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
                    
                    # Set criteria confirmed and move to search
                    st.session_state.criteria_confirmed = True
                    st.session_state.workflow_step = 3  # Move to search/results step
                    
                    # Persist config before potentially long search
                    _save_config_callback()
                    
                    # Continue with search
                    proceed_with_search(st.session_state.final_keywords, user_intent, st.session_state.selected_model)
                else:
                    st.session_state.workflow_step = 1  # Move to keyword confirmation step in assistive mode
                    st.rerun()  # Add rerun to refresh UI and show keyword confirmation step
            else:
                st.warning("LLM did not generate valid keywords.")
                st.session_state.keywords_generated = False

        except ValueError as ve: # Catches API key error from get_llm_response
            st.session_state.error_message = f"Configuration Error: {ve}"
            st.error(st.session_state.error_message)
        except Exception as e:
            st.session_state.error_message = f"An error occurred during keyword generation: {e}"
            st.error(st.session_state.error_message)
            st.exception(e) # Show full traceback for other errors
            # Persist state even on error - REMOVED (handled by on_change)
            # save_config({
            #     'openrouter_api_key': st.session_state.openrouter_api_key,
            #     'huggingface_hub_token': st.session_state.huggingface_hub_token,
            #     'selected_model': st.session_state.selected_model
            # })

# --- Keyword Refinement and Confirmation ---
elif st.session_state.workflow_step == 1:
    # This section should only be reached in assistive mode
    # Add a header showing the current mode
    st.info(f"Current Mode: {st.session_state.mode.capitalize()}")
    
    st.success(f"Generated Keywords: `{', '.join(st.session_state.generated_keywords)}`")
    st.subheader("1a. Refine Keywords & Search")
    st.session_state.refined_keywords_str = st.text_area(
        "Edit, add, or remove keywords (comma-separated):",
        value=st.session_state.refined_keywords_str,
        key='keyword_refinement_area',
        height=75
    )

    if st.button("Confirm Keywords and Continue"):
        final_keywords = [kw.strip() for kw in st.session_state.refined_keywords_str.split(',') if kw.strip()]

        if not final_keywords:
            st.warning("Please provide at least one keyword to search.")
        else:
            st.info(f"Using keywords for search: `{', '.join(final_keywords)}`")
            st.session_state.final_keywords = final_keywords
            st.session_state.workflow_step = 2  # Move to criteria generation step
            # Persist config before moving to next step - REMOVED
            # save_config({
            #     'openrouter_api_key': st.session_state.openrouter_api_key,
            #     'huggingface_hub_token': st.session_state.huggingface_hub_token,
            #     'selected_model': st.session_state.selected_model
            # })
            st.rerun()  # Rerun to update UI

# --- Criteria Generation and Confirmation ---
elif st.session_state.workflow_step == 2:
    # In auto mode, this section should never be reached as we skip directly to step 3
    # But if we somehow get here in auto mode, proceed automatically
    if st.session_state.mode == "auto" and not st.session_state.criteria_confirmed:
        st.info(f"Current Mode: {st.session_state.mode.capitalize()}")
        st.info(f"Using keywords for search: `{', '.join(st.session_state.final_keywords)}`")
        user_intent = st.session_state.user_intent_main_input # Needs to be retrieved correctly
        
        st.subheader("2. Generating Evaluation Criteria...")
        with st.spinner("Asking LLM to generate evaluation criteria based on your intent..."):
            try:
                # Use prompt content from session state
                criteria_prompt_template = st.session_state.criteria_prompt_content
                if not criteria_prompt_template or criteria_prompt_template.startswith("Error:"):
                    raise ValueError("Criteria Generation Prompt is missing or invalid. Check sidebar settings.")
                
                formatted_criteria_prompt = criteria_prompt_template.format(user_intent=user_intent)
                llm_model_to_use = st.session_state.selected_model # Use model from session state
                criteria_llm_output = get_llm_response(prompt=formatted_criteria_prompt, model=llm_model_to_use)
                
                # Parse the criteria (expected JSON list of strings)
                try:
                    # --- Improved Parsing Logic --- 
                    json_str_crit = criteria_llm_output.strip()
                    if json_str_crit.startswith("```json"):
                        json_str_crit = json_str_crit[7:].strip() # Remove ```json
                    if json_str_crit.endswith("```"):
                        json_str_crit = json_str_crit[:-3].strip() # Remove ```
                    
                    # Final strip just in case
                    json_str_crit = json_str_crit.strip()
                    
                    parsed_criteria = json.loads(json_str_crit)
                    # --- End Improved Parsing Logic ---
                    
                    if isinstance(parsed_criteria, list) and all(isinstance(item, str) for item in parsed_criteria):
                        st.session_state.dynamic_criteria = parsed_criteria
                        st.success("Successfully generated evaluation criteria.")
                    else:
                        raise ValueError("LLM output for criteria was not a list of strings.")
                except (json.JSONDecodeError, ValueError) as crit_e:
                    st.warning(f"Could not parse LLM output for criteria as a valid JSON list of strings: {crit_e}")
                    st.text_area("LLM Criteria Output (raw):", criteria_llm_output, height=100, disabled=True)
                    st.info("Proceeding with default criteria. Evaluation might be less specific.")
                    # Ensure fallback is also stored (as a list)
                    st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
                
            except ValueError as ve:
                # Raised if prompt content is bad
                st.error(f"Prompt Error: {ve}")
                raise # Stop execution if prompt is missing
            except Exception as gen_crit_e:
                st.error(f"An error occurred during criteria generation: {gen_crit_e}")
                # Decide if we should proceed with a fallback or stop
                st.info("Proceeding with default criteria due to error. Evaluation might be less specific.")
                # Ensure fallback is also stored (as a list)
                st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
        
        # Set criteria confirmed and move to search
        st.session_state.criteria_confirmed = True
        st.session_state.workflow_step = 3  # Move to search/results step
        
        # Persist config before potentially long search - REMOVED
        # save_config({
        #     'openrouter_api_key': st.session_state.openrouter_api_key,
        #     'huggingface_hub_token': st.session_state.huggingface_hub_token,
        #     'selected_model': st.session_state.selected_model
        # })

        # Continue with search
        proceed_with_search(st.session_state.final_keywords, user_intent, st.session_state.selected_model)
    else:
        # Assistive mode - show criteria editing UI
        st.info(f"Current Mode: {st.session_state.mode.capitalize()}")
        st.info(f"Using keywords for search: `{', '.join(st.session_state.final_keywords)}`")
        user_intent = st.session_state.user_intent_main_input # Get from text area state
        
        # Check if criteria are already generated
        if not st.session_state.criteria_confirmed:
            st.subheader("2. Generating Evaluation Criteria...")
            with st.spinner("Asking LLM to generate evaluation criteria based on your intent..."):
                try:
                    # Use prompt content from session state
                    criteria_prompt_template = st.session_state.criteria_prompt_content
                    if not criteria_prompt_template or criteria_prompt_template.startswith("Error:"):
                        raise ValueError("Criteria Generation Prompt is missing or invalid. Check sidebar settings.")
                    
                    formatted_criteria_prompt = criteria_prompt_template.format(user_intent=user_intent)
                    llm_model_to_use = st.session_state.selected_model # Use model from session state
                    criteria_llm_output = get_llm_response(prompt=formatted_criteria_prompt, model=llm_model_to_use)
                    
                    # Parse the criteria (expected JSON list of strings)
                    try:
                        # --- Improved Parsing Logic --- 
                        json_str_crit = criteria_llm_output.strip()
                        if json_str_crit.startswith("```json"):
                            json_str_crit = json_str_crit[7:].strip() # Remove ```json
                        if json_str_crit.endswith("```"):
                            json_str_crit = json_str_crit[:-3].strip() # Remove ```
                        
                        # Final strip just in case
                        json_str_crit = json_str_crit.strip()
                        
                        parsed_criteria = json.loads(json_str_crit)
                        # --- End Improved Parsing Logic ---
                        
                        if isinstance(parsed_criteria, list) and all(isinstance(item, str) for item in parsed_criteria):
                            st.session_state.dynamic_criteria = parsed_criteria
                            st.success("Successfully generated evaluation criteria.")
                        else:
                            raise ValueError("LLM output for criteria was not a list of strings.")
                    except (json.JSONDecodeError, ValueError) as crit_e:
                        st.warning(f"Could not parse LLM output for criteria as a valid JSON list of strings: {crit_e}")
                        st.text_area("LLM Criteria Output (raw):", criteria_llm_output, height=100, disabled=True)
                        st.info("Proceeding with default criteria. Evaluation might be less specific.")
                        # Ensure fallback is also stored (as a list)
                        st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
                    
                except ValueError as ve:
                    # Raised if prompt content is bad
                    st.error(f"Prompt Error: {ve}")
                    raise # Stop execution if prompt is missing
                except Exception as gen_crit_e:
                    st.error(f"An error occurred during criteria generation: {gen_crit_e}")
                    # Decide if we should proceed with a fallback or stop
                    st.info("Proceeding with default criteria due to error. Evaluation might be less specific.")
                    # Ensure fallback is also stored (as a list)
                    st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
            
            # Display criteria editing UI
            st.subheader("2a. Review and Confirm Evaluation Criteria")
            
            # Add helpful guidance
            with st.expander("ℹ️ Tips for good evaluation criteria", expanded=False):
                st.markdown("""
                * **Be specific** - Criteria should be clear and specific to your research intent
                * **Be measurable** - Each criterion should be something that can be evaluated objectively
                * **Keep them concise** - One sentence per criterion works best
                * **Include domain-specific factors** - Add criteria specific to your domain or task
                """)

            # Initialize or update criteria state
            if not st.session_state.criteria_generated:
                # First time generating criteria
                st.session_state.current_criteria = st.session_state.dynamic_criteria.copy() if st.session_state.dynamic_criteria else [""]
                st.session_state.criteria_generated = True
                st.session_state.criteria_editing_mode = False

            # Create columns for the main actions
            edit_col1, edit_col2 = st.columns([6, 4])
            
            with edit_col1:
                if not st.session_state.criteria_editing_mode:
                    if st.button("✏️ Edit Criteria", type="secondary"):
                        st.session_state.criteria_editing_mode = True
                        st.rerun()

            with edit_col2:
                if not st.session_state.criteria_editing_mode:
                    if st.button("🔄 Regenerate Criteria", type="secondary"):
                        st.session_state.criteria_generated = False
                        st.session_state.current_criteria = None
                        st.rerun()

            # Display mode: Show criteria as read-only with edit button
            if not st.session_state.criteria_editing_mode:
                st.write("### Current Evaluation Criteria")
                for i, criterion in enumerate(st.session_state.current_criteria):
                    if criterion.strip():
                        st.markdown(f"**{i+1}.** {criterion}")
                
                # Confirm button in display mode
                if st.button("✅ Confirm Criteria and Search", type="primary"):
                    # Filter out empty criteria
                    valid_criteria = [c.strip() for c in st.session_state.current_criteria if c.strip()]
                    
                    if not valid_criteria:
                        st.warning("No criteria provided. Using default criterion.")
                        st.session_state.dynamic_criteria = ["Evaluate overall relevance to the user intent."]
                    else:
                        st.session_state.dynamic_criteria = valid_criteria
                        st.success(f"Using {len(valid_criteria)} confirmed criteria for evaluation.")
                    
                    # Set the criteria confirmation flag and update workflow step
                    st.session_state.criteria_confirmed = True
                    st.session_state.workflow_step = 3  # Move to search/results step
                    
                    # Persist config before potentially long search
                    _save_config_callback()
                    
                    # Continue with search
                    proceed_with_search(st.session_state.final_keywords, user_intent, st.session_state.selected_model)

            # Edit mode: Show editable criteria with add/remove functionality
            else:
                st.write("### Edit Evaluation Criteria")
                st.write("Modify, add, or remove criteria:")

                # Store the number of criteria in session state
                if 'num_criteria' not in st.session_state:
                    st.session_state.num_criteria = len(st.session_state.current_criteria)

                # Create a container for criteria
                criteria_container = st.container()
                
                # Function to update criterion
                def update_criterion(i):
                    new_value = st.session_state[f"criterion_{i}"]
                    if i < len(st.session_state.current_criteria):
                        st.session_state.current_criteria[i] = new_value

                # Display existing criteria
                with criteria_container:
                    for i in range(st.session_state.num_criteria):
                        col1, col2 = st.columns([10, 1])
                        with col1:
                            # Get existing value or empty string
                            current_value = st.session_state.current_criteria[i] if i < len(st.session_state.current_criteria) else ""
                            st.text_input(
                                f"Criterion {i+1}",
                                value=current_value,
                                key=f"criterion_{i}",
                                on_change=update_criterion,
                                args=(i,)
                            )
                        with col2:
                            # Only show delete button if we have more than one criterion
                            if st.session_state.num_criteria > 1:
                                if st.button("🗑️", key=f"delete_{i}"):
                                    st.session_state.current_criteria.pop(i)
                                    st.session_state.num_criteria -= 1
                                    st.rerun()

                # Add criterion button
                if st.button("➕ Add Criterion"):
                    st.session_state.current_criteria.append("")
                    st.session_state.num_criteria += 1
                    st.rerun()

                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Save Changes", type="primary"):
                        # Update criteria, removing empty ones
                        st.session_state.current_criteria = [
                            c for c in st.session_state.current_criteria if c.strip()
                        ]
                        if not st.session_state.current_criteria:
                            st.session_state.current_criteria = ["Evaluate overall relevance to the user intent."]
                        st.session_state.criteria_editing_mode = False
                        st.rerun()

                with col2:
                    if st.button("❌ Cancel", type="secondary"):
                        # Restore original criteria
                        st.session_state.current_criteria = st.session_state.dynamic_criteria.copy()
                        st.session_state.criteria_editing_mode = False
                        st.rerun()

                st.markdown("---")
                st.info("Click 'Save Changes' when done editing, or 'Cancel' to discard changes.")

# --- Search and Results Display ---
elif st.session_state.workflow_step == 3:
    user_intent = st.session_state.user_intent_main_input # Get from text area state
    if not st.session_state.search_triggered:
        # If we're in step 3 but search isn't triggered yet, start the search
        # Ensure config is saved before starting
        _save_config_callback()
        proceed_with_search(st.session_state.final_keywords, user_intent, st.session_state.selected_model)
    
    # The rest of the results display logic will run as normal when search_triggered is True

# --- Display Results ---
if st.session_state.search_triggered and not st.session_state.error_message:
    if st.session_state.evaluated_datasets:
        st.subheader("📊 Results") # Added icon
        st.info(f"Current Mode: {st.session_state.mode.capitalize()}")
        df = pd.DataFrame(st.session_state.evaluated_datasets)

        if df.empty:
            st.info("Search found datasets, but none could be successfully evaluated.")
        else:
            # --- Prepare DataFrame for Display (using standardized fields) ---
            display_columns = [
                'url', 'id', 'clear_summary', 'domain', 'task_type', 
                'relevance_score',
                'reasoning',
                'data_size_estimate', 'key_features_columns', 
                'data_quality_hints', 'potential_biases_mentioned', 'license_type'
            ]
            existing_display_columns = [col for col in display_columns if col in df.columns]
            df_display = df[existing_display_columns].copy() # Work on a copy

            # --- Handle List Columns (Make them display nicely) --- 
            list_columns = ['key_features_columns', 'data_quality_hints', 'potential_biases_mentioned']
            for col in list_columns:
                if col in df_display.columns:
                    # Ensure data is list, convert None to empty list, join elements
                    df_display[col] = df_display[col].apply(
                        lambda x: ", ".join(map(str, x)) if isinstance(x, list) else (str(x) if x else "N/A")
                    )
                else:
                    # Add empty column if it was expected but missing, maybe filled with N/A
                    df_display[col] = "N/A" 

            # --- Fill NA for potentially missing optional fields ---
            for col in display_columns:
                if col in df_display.columns and col not in ['url', 'id']:
                    if col == 'relevance_score':
                        df_display[col] = df_display[col].round(3) # Format score
                        df_display[col] = df_display[col].fillna(0.0) # Fill NA score with 0
                    elif col == 'reasoning':
                        df_display[col] = df_display[col].fillna("N/A")
                    else:
                        df_display[col] = df_display[col].fillna("N/A")

            # --- Define Column Order and Rename ---
            # Bring score near the start
            column_order = ['url', 'id', 'relevance_score', 'reasoning'] + [col for col in display_columns if col not in ['url', 'id', 'relevance_score', 'reasoning']]
            final_column_order = [col for col in column_order if col in df_display.columns]
            df_final_display = df_display[final_column_order]

            rename_map = {
                'id': 'ID', 
                'url': 'Link',
                'clear_summary': 'Summary (LLM)', 
                'domain': 'Domain',
                'task_type': 'Task',
                'relevance_score': 'Relevance',
                'reasoning': 'Reasoning (LLM)',
                'data_size_estimate': 'Size Est.',
                'key_features_columns': 'Key Features/Columns',
                'data_quality_hints': 'Quality Hints',
                'potential_biases_mentioned': 'Potential Biases',
                'license_type': 'License'
            }
            # Only rename columns that actually exist in the final display DataFrame
            rename_map_existing = {k: v for k, v in rename_map.items() if k in df_final_display.columns}
            df_final_display.rename(columns=rename_map_existing, inplace=True)
            
            # --- Define function to convert df to csv for download --- 
            @st.cache_data # Cache the conversion
            def convert_df_to_csv(df_to_convert):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(df_final_display)

            # --- Display Ranked Table (Previously Tab 1) ---
            st.markdown("#### Ranked Datasets")
            # Add download button for the table
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name='adrs_evaluated_datasets.csv',
                mime='text/csv',
            )
            # Display Evaluation Errors (if any)
            if st.session_state.evaluation_errors:
                with st.expander("⚠️ View Evaluation Issues"):
                    for error in st.session_state.evaluation_errors:
                        st.warning(error)
            # Display the dataframe
            st.dataframe(
                df_final_display,
                hide_index=True,
                column_config={
                    "ID": st.column_config.TextColumn("ID"),
                    "Link": st.column_config.LinkColumn(
                        "Link",
                        help="Click to visit the Hugging Face dataset page",
                        display_text="Visit ↗",
                        validate="^https?://.*"
                    ),
                    # Use the renamed columns from rename_map_existing
                    rename_map_existing.get('clear_summary', 'Summary (LLM)'): st.column_config.TextColumn(rename_map_existing.get('clear_summary', 'Summary (LLM)'), width="medium"),
                    rename_map_existing.get('domain', 'Domain'): st.column_config.TextColumn(rename_map_existing.get('domain', 'Domain'), width="small"),
                    rename_map_existing.get('task_type', 'Task'): st.column_config.TextColumn(rename_map_existing.get('task_type', 'Task'), width="small"),
                    rename_map_existing.get('relevance_score', 'Relevance'): st.column_config.NumberColumn(format="%.3f", width="small"),
                    rename_map_existing.get('reasoning', 'Reasoning (LLM)'): st.column_config.TextColumn(rename_map_existing.get('reasoning', 'Reasoning (LLM)'), width="large"),
                    rename_map_existing.get('data_size_estimate', 'Size Est.'): st.column_config.TextColumn(rename_map_existing.get('data_size_estimate', 'Size Est.'), width="small"),
                    rename_map_existing.get('key_features_columns', 'Key Features/Columns'): st.column_config.TextColumn(rename_map_existing.get('key_features_columns', 'Key Features/Columns'), width="medium"),
                    rename_map_existing.get('data_quality_hints', 'Quality Hints'): st.column_config.TextColumn(rename_map_existing.get('data_quality_hints', 'Quality Hints'), width="medium"),
                    rename_map_existing.get('potential_biases_mentioned', 'Potential Biases'): st.column_config.TextColumn(rename_map_existing.get('potential_biases_mentioned', 'Potential Biases'), width="medium"),
                    rename_map_existing.get('license_type', 'License'): st.column_config.TextColumn(rename_map_existing.get('license_type', 'License'), width="small"),
                },
                use_container_width=True
            )
            
            st.markdown("--- ") # Add a separator between table and report section

            # --- Display Report Section (Previously Tab 2) ---
            st.markdown("#### Synthesis Report")
            
            # Check if in auto mode and report not already generated
            if st.session_state.mode == "auto" and not st.session_state.generated_report:
                # Automatically generate report in auto mode
                st.info("Auto-generating report and visualizations...")
                
                # --- Prepare context for the report prompt ---
                user_intent = st.session_state.user_intent_main_input # Get from text area state
                report_context = f"""
User Intent:
{user_intent}

Keywords Used:
{st.session_state.refined_keywords_str}

Evaluation Criteria Generated:
{json.dumps(st.session_state.dynamic_criteria, indent=2)}

Evaluated Datasets Summary (Top {len(st.session_state.evaluated_datasets)}):
{df_final_display.to_markdown(index=False)} 
""" 

                # --- Define the UPDATED Report Generation Prompt --- 
                # Use prompt content from session state
                report_prompt_template = st.session_state.report_prompt_content
                if not report_prompt_template or report_prompt_template.startswith("Error:"):
                    raise ValueError("Report Generation Prompt is missing or invalid. Check sidebar settings or create prompts/generate_report.txt.")
                
                # Format the prompt (ensure it contains {report_context} and {researcher_profile})
                report_prompt = report_prompt_template.format(
                    report_context=report_context,
                    researcher_profile=st.session_state.get('researcher_profile', 'No profile provided')
                )

                try:
                    # --- Generate Charts First (using Streamlit native + Altair) --- 
                    charts = {}
                    
                    # 1. Relevance Score Distribution (using Altair)
                    if 'Relevance' in df_final_display.columns:
                        score_data_df = df_final_display[['ID', 'Relevance']].dropna().reset_index()
                        if not score_data_df.empty:
                            if len(score_data_df) < 30: # Plot individual scores if not too many
                                score_chart = alt.Chart(score_data_df).mark_bar().encode(
                                    x=alt.X('ID', sort='-y', title="Dataset ID"), # Sort by relevance
                                    y=alt.Y('Relevance', title="Relevance Score"),
                                    tooltip=['ID', 'Relevance']
                                ).properties(
                                    title='Relevance Score per Dataset'
                                ).interactive() # Enable interactivity
                                charts['relevance_scores'] = score_chart
                            else: # Otherwise, create a histogram
                                score_chart = alt.Chart(score_data_df).mark_bar().encode(
                                    alt.X("Relevance", bin=alt.Bin(maxbins=10), title="Relevance Score Bins"), # Bin scores
                                    alt.Y('count()', title="Number of Datasets"),
                                    tooltip=[alt.Tooltip("Relevance", bin=alt.Bin(maxbins=10)), alt.Tooltip('count()')]
                                ).properties(
                                    title='Distribution of Relevance Scores'
                                ).interactive()
                                charts['relevance_scores'] = score_chart
                                
                    # 2. Domain Counts (using Altair)
                    if 'Domain' in df_final_display.columns:
                        domain_counts_df = df_final_display['Domain'].value_counts().head(15).reset_index() # Top 15
                        domain_counts_df.columns = ['Domain', 'Count'] # Rename for Altair
                        if not domain_counts_df.empty:
                            domain_chart = alt.Chart(domain_counts_df).mark_bar().encode(
                                x=alt.X('Domain', sort='-y', title="Dataset Domain"),
                                y=alt.Y('Count', title="Number of Datasets"),
                                tooltip=['Domain', 'Count']
                            ).properties(
                                title='Top 15 Dataset Domains'
                            ).interactive()
                            charts['domain_counts'] = domain_chart

                    # --- NEW: 3. Task Type Counts (using Altair) ---
                    task_col_name = rename_map_existing.get('task_type', 'Task') # Get potentially renamed column
                    if task_col_name in df_final_display.columns:
                        task_counts_df = df_final_display[task_col_name].value_counts().head(15).reset_index()
                        task_counts_df.columns = ['Task', 'Count'] # Rename for Altair
                        if not task_counts_df.empty:
                            task_chart = alt.Chart(task_counts_df).mark_bar().encode(
                                x=alt.X('Task', sort='-y', title="Primary ML Task"),
                                y=alt.Y('Count', title="Number of Datasets"),
                                tooltip=['Task', 'Count']
                            ).properties(
                                title='Top 15 Dataset Task Types'
                            ).interactive()
                            charts['task_counts'] = task_chart

                    # --- NEW: 4. License Type Counts (using Altair) ---
                    license_col_name = rename_map_existing.get('license_type', 'License') # Get potentially renamed column
                    if license_col_name in df_final_display.columns:
                        license_counts_df = df_final_display[license_col_name].value_counts().head(15).reset_index()
                        license_counts_df.columns = ['License', 'Count']
                        if not license_counts_df.empty:
                            license_chart = alt.Chart(license_counts_df).mark_bar().encode(
                                x=alt.X('License', sort='-y', title="License Type"),
                                y=alt.Y('Count', title="Number of Datasets"),
                                tooltip=['License', 'Count']
                            ).properties(
                                title='Top 15 Dataset Licenses'
                            ).interactive()
                            charts['license_counts'] = license_chart

                    # --- NEW: 5. Data Size Estimate Counts (using Altair) ---
                    size_col_name = rename_map_existing.get('data_size_estimate', 'Size Est.') # Get potentially renamed column
                    if size_col_name in df_final_display.columns:
                        # Define a reasonable order for size categories if possible
                        size_order = ['Small (<1k rows/samples)', 'Medium (1k-100k)', 'Large (100k-1M)', 'Very Large (>1M)', 'N/A', 'Unknown']
                        size_counts_df = df_final_display[size_col_name].value_counts().reset_index()
                        size_counts_df.columns = ['Size Estimate', 'Count']
                        if not size_counts_df.empty:
                            size_chart = alt.Chart(size_counts_df).mark_bar().encode(
                                x=alt.X('Size Estimate', sort=size_order, title="Estimated Size"), # Apply sorting if categories match
                                y=alt.Y('Count', title="Number of Datasets"),
                                tooltip=['Size Estimate', 'Count']
                            ).properties(
                                title='Dataset Size Estimates'
                            ).interactive()
                            charts['size_counts'] = size_chart

                    # --- NEW: 6. Score vs Popularity (Downloads) Scatter Plot (using Altair) ---
                    # Use the original df DataFrame here as it has raw downloads/likes/score
                    scatter_cols = ['id', 'relevance_score', 'downloads']
                    if all(col in df.columns for col in scatter_cols):
                        scatter_data = df[scatter_cols].copy()
                        # Handle potential missing values for plotting
                        scatter_data.dropna(subset=['relevance_score', 'downloads'], inplace=True)
                        # Convert downloads to numeric, coercing errors
                        scatter_data['downloads'] = pd.to_numeric(scatter_data['downloads'], errors='coerce')
                        scatter_data.dropna(subset=['downloads'], inplace=True) # Drop rows where conversion failed
                        scatter_data['downloads'] = scatter_data['downloads'].astype(int) # Ensure integer

                        if not scatter_data.empty:
                            # Define tooltips using original column names from df
                            tooltips = [
                                alt.Tooltip('id', title='ID'),
                                alt.Tooltip('relevance_score', title='Relevance Score', format='.3f'),
                                alt.Tooltip('downloads', title='Downloads')
                            ]
                            # Add likes to tooltip if available
                            if 'likes' in df.columns:
                                scatter_data['likes'] = pd.to_numeric(df.loc[scatter_data.index, 'likes'], errors='coerce').fillna(0).astype(int)
                                tooltips.append(alt.Tooltip('likes', title='Likes'))

                            scatter_plot = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.7).encode(
                                x=alt.X('relevance_score', title='Relevance Score (LLM)', scale=alt.Scale(zero=False)),
                                y=alt.Y('downloads', title='Downloads', scale=alt.Scale(type='log', base=10, zero=False), axis=alt.Axis(format=',d')), # Use log scale for downloads
                                tooltip=tooltips
                            ).properties(
                                title='Relevance Score vs. Downloads (Log Scale)'
                            ).interactive() # Enable zooming and panning
                            charts['score_vs_popularity'] = scatter_plot

                    st.session_state.report_charts = charts

                    # --- Generate Text Report (using LLM) --- 
                    with st.spinner("Synthesizing findings with LLM..."):
                        llm_model_to_use = st.session_state.selected_model # Use model from session state
                        report_output = get_llm_response(
                            prompt=report_prompt, 
                            model=llm_model_to_use,
                            # Increase max tokens if needed for longer reports
                            # max_tokens=4000 
                        )
                        st.session_state.generated_report = report_output
                        st.success("Report and visualizations generated successfully!")

                except Exception as report_e:
                    st.error(f"An error occurred during report generation: {report_e}")
                    st.exception(report_e) # Show details
                    st.session_state.generated_report = f"Error generating report: {report_e}" # Store error message
                    st.session_state.report_charts = None # Clear charts on error
            
            # In assistive mode, provide button to generate report
            elif st.session_state.mode == "assistive":
                # --- Report Generation Section (Moved inside tab2) ---
                st.markdown("Click the button below to generate visualizations and a detailed report synthesizing the findings.")

                # Add a container for the button to help with CSS hiding for print
                report_button_container = st.container()
                with report_button_container:
                    if st.button("Generate Full Report & Visualizations", key="generate_report_button"):
                        st.session_state.generated_report = None # Clear previous report
                        st.session_state.report_charts = None    # Clear previous charts

                        # --- Prepare context for the report prompt (same as before) --- 
                        user_intent = st.session_state.user_intent_main_input # Get from text area state
                        report_context = f"""
User Intent:
{user_intent}

Keywords Used:
{st.session_state.refined_keywords_str}

Evaluation Criteria Generated:
{json.dumps(st.session_state.dynamic_criteria, indent=2)}

Evaluated Datasets Summary (Top {len(st.session_state.evaluated_datasets)}):
{df_final_display.to_markdown(index=False)} 
""" 

                        # --- Define the UPDATED Report Generation Prompt --- 
                        # Use prompt content from session state
                        report_prompt_template = st.session_state.report_prompt_content
                        if not report_prompt_template or report_prompt_template.startswith("Error:"):
                            raise ValueError("Report Generation Prompt is missing or invalid. Check sidebar settings or create prompts/generate_report.txt.")
                        
                        # Format the prompt (ensure it contains {report_context} and {researcher_profile})
                        report_prompt = report_prompt_template.format(
                            report_context=report_context,
                            researcher_profile=st.session_state.get('researcher_profile', 'No profile provided')
                        )

                        try:
                            st.info("Generating report and visualizations... This may take a moment.")
                            
                            # --- Generate Charts First (using Streamlit native + Altair) --- 
                            charts = {}
                            
                            # 1. Relevance Score Distribution (using Altair)
                            if 'Relevance' in df_final_display.columns:
                                score_data_df = df_final_display[['ID', 'Relevance']].dropna().reset_index()
                                if not score_data_df.empty:
                                    if len(score_data_df) < 30: # Plot individual scores if not too many
                                        score_chart = alt.Chart(score_data_df).mark_bar().encode(
                                            x=alt.X('ID', sort='-y', title="Dataset ID"), # Sort by relevance
                                            y=alt.Y('Relevance', title="Relevance Score"),
                                            tooltip=['ID', 'Relevance']
                                        ).properties(
                                            title='Relevance Score per Dataset'
                                        ).interactive() # Enable interactivity
                                        charts['relevance_scores'] = score_chart
                                    else: # Otherwise, create a histogram
                                        score_chart = alt.Chart(score_data_df).mark_bar().encode(
                                            alt.X("Relevance", bin=alt.Bin(maxbins=10), title="Relevance Score Bins"), # Bin scores
                                            alt.Y('count()', title="Number of Datasets"),
                                            tooltip=[alt.Tooltip("Relevance", bin=alt.Bin(maxbins=10)), alt.Tooltip('count()')]
                                        ).properties(
                                            title='Distribution of Relevance Scores'
                                        ).interactive()
                                        charts['relevance_scores'] = score_chart
                                        
                            # 2. Domain Counts (using Altair)
                            if 'Domain' in df_final_display.columns:
                                domain_counts_df = df_final_display['Domain'].value_counts().head(15).reset_index() # Top 15
                                domain_counts_df.columns = ['Domain', 'Count'] # Rename for Altair
                                if not domain_counts_df.empty:
                                    domain_chart = alt.Chart(domain_counts_df).mark_bar().encode(
                                        x=alt.X('Domain', sort='-y', title="Dataset Domain"),
                                        y=alt.Y('Count', title="Number of Datasets"),
                                        tooltip=['Domain', 'Count']
                                    ).properties(
                                        title='Top 15 Dataset Domains'
                                    ).interactive()
                                    charts['domain_counts'] = domain_chart

                            # --- NEW: 3. Task Type Counts (using Altair) ---
                            task_col_name = rename_map_existing.get('task_type', 'Task') # Get potentially renamed column
                            if task_col_name in df_final_display.columns:
                                task_counts_df = df_final_display[task_col_name].value_counts().head(15).reset_index()
                                task_counts_df.columns = ['Task', 'Count'] # Rename for Altair
                                if not task_counts_df.empty:
                                    task_chart = alt.Chart(task_counts_df).mark_bar().encode(
                                        x=alt.X('Task', sort='-y', title="Primary ML Task"),
                                        y=alt.Y('Count', title="Number of Datasets"),
                                        tooltip=['Task', 'Count']
                                    ).properties(
                                        title='Top 15 Dataset Task Types'
                                    ).interactive()
                                    charts['task_counts'] = task_chart

                            # --- NEW: 4. License Type Counts (using Altair) ---
                            license_col_name = rename_map_existing.get('license_type', 'License') # Get potentially renamed column
                            if license_col_name in df_final_display.columns:
                                license_counts_df = df_final_display[license_col_name].value_counts().head(15).reset_index()
                                license_counts_df.columns = ['License', 'Count']
                                if not license_counts_df.empty:
                                    license_chart = alt.Chart(license_counts_df).mark_bar().encode(
                                        x=alt.X('License', sort='-y', title="License Type"),
                                        y=alt.Y('Count', title="Number of Datasets"),
                                        tooltip=['License', 'Count']
                                    ).properties(
                                        title='Top 15 Dataset Licenses'
                                    ).interactive()
                                    charts['license_counts'] = license_chart

                            # --- NEW: 5. Data Size Estimate Counts (using Altair) ---
                            size_col_name = rename_map_existing.get('data_size_estimate', 'Size Est.') # Get potentially renamed column
                            if size_col_name in df_final_display.columns:
                                # Define a reasonable order for size categories if possible
                                size_order = ['Small (<1k rows/samples)', 'Medium (1k-100k)', 'Large (100k-1M)', 'Very Large (>1M)', 'N/A', 'Unknown']
                                size_counts_df = df_final_display[size_col_name].value_counts().reset_index()
                                size_counts_df.columns = ['Size Estimate', 'Count']
                                if not size_counts_df.empty:
                                    size_chart = alt.Chart(size_counts_df).mark_bar().encode(
                                        x=alt.X('Size Estimate', sort=size_order, title="Estimated Size"), # Apply sorting if categories match
                                        y=alt.Y('Count', title="Number of Datasets"),
                                        tooltip=['Size Estimate', 'Count']
                                    ).properties(
                                        title='Dataset Size Estimates'
                                    ).interactive()
                                    charts['size_counts'] = size_chart

                            # --- NEW: 6. Score vs Popularity (Downloads) Scatter Plot (using Altair) ---
                            # Use the original df DataFrame here as it has raw downloads/likes/score
                            scatter_cols = ['id', 'relevance_score', 'downloads']
                            if all(col in df.columns for col in scatter_cols):
                                scatter_data = df[scatter_cols].copy()
                                # Handle potential missing values for plotting
                                scatter_data.dropna(subset=['relevance_score', 'downloads'], inplace=True)
                                # Convert downloads to numeric, coercing errors
                                scatter_data['downloads'] = pd.to_numeric(scatter_data['downloads'], errors='coerce')
                                scatter_data.dropna(subset=['downloads'], inplace=True) # Drop rows where conversion failed
                                scatter_data['downloads'] = scatter_data['downloads'].astype(int) # Ensure integer

                                if not scatter_data.empty:
                                    # Define tooltips using original column names from df
                                    tooltips = [
                                        alt.Tooltip('id', title='ID'),
                                        alt.Tooltip('relevance_score', title='Relevance Score', format='.3f'),
                                        alt.Tooltip('downloads', title='Downloads')
                                    ]
                                    # Add likes to tooltip if available
                                    if 'likes' in df.columns:
                                        scatter_data['likes'] = pd.to_numeric(df.loc[scatter_data.index, 'likes'], errors='coerce').fillna(0).astype(int)
                                        tooltips.append(alt.Tooltip('likes', title='Likes'))

                                    scatter_plot = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.7).encode(
                                        x=alt.X('relevance_score', title='Relevance Score (LLM)', scale=alt.Scale(zero=False)),
                                        y=alt.Y('downloads', title='Downloads', scale=alt.Scale(type='log', base=10, zero=False), axis=alt.Axis(format=',d')), # Use log scale for downloads
                                        tooltip=tooltips
                                    ).properties(
                                        title='Relevance Score vs. Downloads (Log Scale)'
                                    ).interactive() # Enable zooming and panning
                                    charts['score_vs_popularity'] = scatter_plot

                            st.session_state.report_charts = charts

                            # --- Generate Text Report (using LLM) --- 
                            with st.spinner("Synthesizing findings with LLM..."):
                                llm_model_to_use = st.session_state.selected_model # Use model from session state
                                report_output = get_llm_response(
                                    prompt=report_prompt, 
                                    model=llm_model_to_use,
                                    # Increase max tokens if needed for longer reports
                                    # max_tokens=4000 
                                )
                                st.session_state.generated_report = report_output
                                st.success("Report and visualizations generated successfully!")

                        except Exception as report_e:
                            st.error(f"An error occurred during report generation: {report_e}")
                            st.exception(report_e) # Show details
                            st.session_state.generated_report = f"Error generating report: {report_e}" # Store error message
                            st.session_state.report_charts = None # Clear charts on error

            # --- Display Report and Charts (Moved inside tab2) --- 
            if st.session_state.generated_report or st.session_state.report_charts:
                st.markdown("--- ") # Separator

                # Display Text Report if it exists
                if st.session_state.generated_report:
                    # This subheader might be redundant now, consider removing if needed
                    # st.subheader("Full Report") 
                    # --- New logic to intersperse charts and text --- 
                    report_content = st.session_state.generated_report
                    chart_keys_in_report = list(st.session_state.report_charts.keys())
                    
                    # Split the report by chart placeholders
                    # import re # No longer needed here if imported at top
                    # Regex to find placeholders like [CHART:key_name]
                    placeholder_pattern = r"\[CHART:(\w+)\]" # Corrected regex pattern
                    
                    parts = re.split(placeholder_pattern, report_content)
                    
                    # Display the first part (text before any chart)
                    if parts[0]:
                        st.markdown(parts[0])
                    
                    # Iterate through the matches (key, text_after)
                    i = 1
                    while i < len(parts):
                        chart_key = parts[i]
                        text_after = parts[i+1] if (i+1) < len(parts) else ""
                        
                        # Display the chart if the key is valid
                        if chart_key in st.session_state.report_charts:
                            st.altair_chart(st.session_state.report_charts[chart_key], use_container_width=True)
                        else:
                            # If key is invalid or chart missing, show a warning or the placeholder text
                            st.warning(f"⚠️ Chart placeholder found `[CHART:{chart_key}]`, but no corresponding chart was generated.")
                            
                        # Display the text that came after this chart placeholder
                        if text_after:
                            st.markdown(text_after)
                            
                        i += 2 # Move to the next potential key
                    # --- End new logic --- 
                    
                    # --- PDF Note Instead of Print Button --- 
                    st.info("**To save as PDF:** Use your browser's print function (Ctrl+P or Cmd+P) and select 'Save as PDF' as the destination.")
                    # --- End PDF Note ---


    # If search triggered, no error, but ranked datasets list is empty
    elif not st.session_state.error_message and not st.session_state.evaluated_datasets:
        # Check if original search found anything
        if st.session_state.datasets: # Original search found items, but evaluation failed/yielded nothing
            st.warning("Search found datasets, but none could be successfully evaluated by the LLM. Check logs or LLM configuration.")
        else: # Original search found nothing
            st.info("Search completed, but no datasets matching the keywords were found on Hugging Face Hub.")

# --- Report Generation Section ---
# This section is now MOVED inside the Results display logic within tab2
# if st.session_state.evaluated_datasets and not st.session_state.error_message:
#     st.subheader("7. Generate Discovery Report")
#             st.session_state.report_charts = None # Clear charts on error

# # --- Display Report and Charts --- 
# if st.session_state.generated_report:
#     st.markdown(st.session_state.generated_report)


# Display errors that might have occurred during keyword generation or search if not already shown
elif st.session_state.error_message and not st.session_state.search_triggered : # Show generation errors if search wasn't even triggered
    # Error is already displayed where it occurs, this check is redundant if st.error is used above
    pass


# Add a footer or separator
st.markdown("--- ")
# Create a footer with logo and text
footer_col1, footer_col2 = st.columns([1, 8])
with footer_col1:
    st.image("assets/aurak-logo.png", width=200)
with footer_col2:
    st.caption("ADRS is a research tool developed by American University of Ras Al Khaimah, Computer Science Department")