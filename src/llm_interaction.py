import os
import litellm
import json
import datetime
import re # Added import for regex
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
from .metadata_schema import EvaluatedMetadata

# Load environment variables from .env file if it exists
load_dotenv()

# Configure LiteLLM base (optional, often default)
# litellm.api_base = "https://openrouter.ai/api/v1" 

# Set a default model (can be overridden)
DEFAULT_MODEL = "meta-llama/llama-4-maverick" # Updated default model

def get_llm_response(prompt: str, model: str = DEFAULT_MODEL, system_prompt: str = None) -> str:
    """
    Sends a prompt to the specified LLM via LiteLLM/OpenRouter and returns the response.
    Assumes litellm.api_key has been set beforehand.

    Args:
        prompt: The user's prompt.
        model: The OpenRouter model string (e.g., "mistralai/mistral-7b-instruct-v0.2").
        system_prompt: An optional system message to guide the LLM's behavior.

    Returns:
        The text content of the LLM's response.

    Raises:
        ValueError: If litellm.api_key is not set.
        Exception: If the API call fails.
    """
    if not litellm.api_key:
        raise ValueError("OpenRouter API Key not configured. Please set it in the sidebar settings.")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        print(f"--- Sending prompt to {model} via OpenRouter ---")
        # print(f"System Prompt: {system_prompt}") # Less verbose logging
        # print(f"User Prompt: {prompt}")

        response = litellm.completion(
            model=f"openrouter/{model}", # Prefix with "openrouter/"
            messages=messages,
            # api_base="https://openrouter.ai/api/v1" # Can be set here too
        )

        content = response.choices[0].message.content
        # print(f"--- Received response ---") # Less verbose logging
        # print(content)
        # print("------------------------")
        return content.strip()

    except Exception as e:
        # --- Enhanced Error Logging ---
        import traceback
        print(f"Error calling LiteLLM API for model {model}. Error Type: {type(e).__name__}, Error: {e}")
        print("Full Traceback:")
        traceback.print_exc() # Print full traceback to console
        # --- End Enhanced Error Logging ---
        # Handle specific exceptions like RateLimitError, APIConnectionError if needed
        raise # Re-raise the exception so it can be caught upstream if needed

# Example usage removed as API key is not set here by default 

def datetime_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def evaluate_dataset_with_llm(
    raw_metadata: Dict,
    user_intent: str,
    dynamic_criteria: Union[List[str], str],
    evaluation_prompt_template: str, # Pass the template content directly
    model: str = DEFAULT_MODEL,
    researcher_profile: str = "No profile provided"
) -> Optional[EvaluatedMetadata]:
    """
    Analyzes raw Hugging Face dataset metadata using an LLM based on user intent AND
    pre-generated dynamic criteria. Extracts key fields, assigns a relevance score,
    and provides reasoning based on the criteria.

    Args:
        raw_metadata: A dictionary containing the raw metadata fetched from Hugging Face.
        user_intent: The user's research intent string.
        dynamic_criteria: The list of criteria strings generated based on the user intent,
                          or a pre-formatted string representation (e.g., JSON list).
        evaluation_prompt_template: The string template for the evaluation prompt.
        model: The OpenRouter model string to use for analysis.
        researcher_profile: Optional string containing the researcher's background and interests.

    Returns:
        A dictionary conforming to the LLM-generated parts of the EvaluatedMetadata 
        schema (score, reasoning, summary, etc.), or None if an error occurs.
        Fields like id, url, downloads, likes need to be added *after* this call.
    """
    # Use the provided template string directly
    prompt_template = evaluation_prompt_template

    # Format the raw metadata nicely for the prompt
    try:
        raw_metadata_str = json.dumps(raw_metadata, indent=2, default=datetime_serializer)
    except Exception as json_err:
        print(f"Error serializing raw metadata to JSON: {json_err}")
        raw_metadata_str = str(raw_metadata) # Fallback
        
    # --- Format criteria for prompt --- 
    if isinstance(dynamic_criteria, list):
        # Convert list to a JSON string representation for the prompt
        dynamic_criteria_str = json.dumps(dynamic_criteria, indent=2)
    elif isinstance(dynamic_criteria, str):
        # Assume it's already formatted correctly (e.g., a JSON string)
        dynamic_criteria_str = dynamic_criteria
    else:
        print("Error: dynamic_criteria must be a list of strings or a formatted string.")
        return None
    
    # --- Debugging: Print parts of the prompt --- 
    print(f"--- DEBUG: Evaluating dataset {raw_metadata.get('id', '[unknown ID]')} ---")
    # print(f"User Intent: {user_intent}") # Can be long
    print(f"Dynamic Criteria (as passed to format):\n{dynamic_criteria_str}")
    # Format prompt with intent, metadata, and dynamic criteria
    formatted_prompt = prompt_template.format(
        user_intent=user_intent,
        dynamic_criteria=dynamic_criteria_str,
        raw_metadata=raw_metadata_str,
        researcher_profile=researcher_profile
    )

    # --- Debugging: Print full prompt --- 
    print(f"--- DEBUG: Full prompt being sent for {raw_metadata.get('id', '[unknown ID]')} ---")
    # Truncate if too long for console? Maybe print first/last 1000 chars
    print(formatted_prompt[:1000] + "..." + formatted_prompt[-1000:])
    print("----------------------------------------------------------")
    # --- End Debugging ---

    try:
        llm_output_str = get_llm_response(prompt=formatted_prompt, model=model)

        # --- Debugging: Print raw LLM output --- 
        print(f"--- DEBUG: Raw LLM output received for {raw_metadata.get('id', '[unknown ID]')} ---")
        print(llm_output_str)
        print("----------------------------------------------------------")
        # --- End Debugging ---

        # The prompt asks for JSON, attempt to parse it
        try:
            # --- Stage 1: Extract potential JSON block --- 
            llm_output_cleaned = llm_output_str.strip()
            json_str = None
            # Try extracting from ```json ... ``` fences first
            match = re.search(r"```json\n(.*?)\n```", llm_output_cleaned, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                print(f"--- DEBUG: Extracted JSON from ```json fence for {raw_metadata.get('id', '[unknown ID]')}")
            else:
                # Fallback: Find the outermost curly braces
                start_index = llm_output_cleaned.find('{')
                end_index = llm_output_cleaned.rfind('}')
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    json_str = llm_output_cleaned[start_index : end_index + 1]
                    print(f"--- DEBUG: Extracted JSON using outer braces for {raw_metadata.get('id', '[unknown ID]')}")
                else:
                    # If no clear block found, use the whole cleaned output as a last resort
                    print(f"Warning: Could not reliably find JSON block for {raw_metadata.get('id', '[unknown ID]')}. Using entire cleaned output.")
                    json_str = llm_output_cleaned
            
            if not json_str:
                print(f"Error: Could not extract any potential JSON content for {raw_metadata.get('id', '[unknown ID]')}.")
                return None

            # --- Stage 2: Clean the extracted string --- 
            json_str_cleaned = json_str
            try:
                # Remove single-line comments (// ...)
                json_str_cleaned = re.sub(r"//.*", "", json_str_cleaned)
                # Remove multi-line comments (/* ... */)
                json_str_cleaned = re.sub(r"/\*.*?\*/", "", json_str_cleaned, flags=re.DOTALL)
                # Remove trailing comma before final brace
                json_str_cleaned = re.sub(r",\s*\}\s*$", "}", json_str_cleaned).strip()
                # Standardize None/True/False
                json_str_cleaned = re.sub(r"\bNone\b", "null", json_str_cleaned)
                json_str_cleaned = re.sub(r"\bTrue\b", "true", json_str_cleaned)
                json_str_cleaned = re.sub(r"\bFalse\b", "false", json_str_cleaned)
            except Exception as clean_err:
                print(f"Warning: Error during JSON cleaning regex for {raw_metadata.get('id', '[unknown ID]')}: {clean_err}. Attempting parse with partially cleaned string.")
                # Proceed with potentially partially cleaned string

            print(f"--- DEBUG: Attempting to parse cleaned JSON snippet for {raw_metadata.get('id', '[unknown ID]')}: {json_str_cleaned[:100]}...{json_str_cleaned[-100:]}")

            # --- Stage 3: Parse the Cleaned JSON String --- 
            evaluated_data: EvaluatedMetadata = json.loads(json_str_cleaned)
            
            # Basic validation (can be expanded)
            if not isinstance(evaluated_data, dict):
                 print(f"Error: LLM output parsed but is not a dictionary: {evaluated_data}")
                 return None
                 
            # Add type validation for score
            if 'relevance_score' in evaluated_data and not isinstance(evaluated_data['relevance_score'], (int, float)):
                print(f"Warning: LLM provided non-numeric relevance_score: {evaluated_data['relevance_score']}. Setting to 0.0.")
                evaluated_data['relevance_score'] = 0.0 # Default score on type error
            elif 'relevance_score' not in evaluated_data:
                 print(f"Warning: LLM did not provide relevance_score. Setting to 0.0.")
                 evaluated_data['relevance_score'] = 0.0 # Default score if missing

            print(f"--- Successfully evaluated metadata for dataset {raw_metadata.get('id', '[unknown ID]')} ---")
            # print(json.dumps(evaluated_data, indent=2)) # Verbose logging if needed
            return evaluated_data

        except json.JSONDecodeError as json_e:
            print(f"Error: Failed to parse LLM output as JSON for dataset {raw_metadata.get('id', '[unknown ID]')}.")
            print(f"LLM Output was:\n---\n{llm_output_str}\n---")
            print(f"JSON Decode Error: {json_e}")
            return None
        except Exception as parse_e:
            print(f"Error processing LLM response for dataset {raw_metadata.get('id', '[unknown ID]')}: {parse_e}")
            print(f"LLM Output was:\n---\n{llm_output_str}\n---")
            return None

    except Exception as llm_e:
        # Error during the LLM call itself (handled in get_llm_response but catch here too)
        print(f"Error during LLM evaluation call for dataset {raw_metadata.get('id', '[unknown ID]')}: {llm_e}")
        return None 