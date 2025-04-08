import argparse
import string # Import string for punctuation removal
import re # Import regex
from src.llm_interaction import get_llm_response
from src.hf_search import search_datasets # Import the search function

def main():
    parser = argparse.ArgumentParser(description="Find Hugging Face datasets based on research intent.")
    parser.add_argument("intent", type=str, help="Describe the type of dataset you are looking for.")
    parser.add_argument("--model", type=str, default=None, help="Optional: Specify the OpenRouter model to use.")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum number of datasets to display.")

    args = parser.parse_args()

    print(f"User Intent: {args.intent}")

    # --- Step 1: Generate Keywords using LLM ---
    print("\n--- Generating Keywords --- ")
    keywords_prompt = f"Based on the research intent '{args.intent}', generate a list of 5-10 relevant keywords for searching Hugging Face datasets. IMPORTANT: Output ONLY the comma-separated list of keywords and nothing else. Do not include introductory text, numbering, or explanations."
    llm_model_to_use = args.model if args.model else None

    try:
        raw_llm_response = get_llm_response(keywords_prompt, model=llm_model_to_use if llm_model_to_use else "mistralai/mistral-7b-instruct-v0.2")

        # --- Improved Keyword Cleaning ---
        # 1. Split into lines and take the last non-empty line
        lines = [line.strip() for line in raw_llm_response.strip().split('\n') if line.strip()]
        keyword_string = lines[-1] if lines else ""

        # Define punctuation to remove (excluding underscore)
        punctuation_to_remove = string.punctuation.replace("_", "")
        translator = str.maketrans('', '', punctuation_to_remove)

        keywords = [
            kw.translate(translator).strip()
            for kw in keyword_string.split(',')
            if kw.translate(translator).strip()
        ]
        # 3. Basic sanity check (remove overly long strings that might be leftover sentences)
        keywords = [kw for kw in keywords if len(kw.split()) < 5] 

        print(f"Cleaned Keywords (underscores preserved): {keywords}")

        if not keywords:
            print("LLM did not generate any valid keywords after cleaning. Exiting.")
            return

        # --- Step 2: Search Hugging Face Hub --- 
        print("\n--- Searching Hugging Face Hub --- ")
        found_datasets = search_datasets(keywords)

        # --- Step 3: Display Results --- 
        if found_datasets:
            print("\n--- Top Dataset Results ---")
            for i, ds in enumerate(found_datasets[:args.max_results]):
                print(f"{i+1}. ID: {ds['id']}")
                print(f"   Author: {ds.get('author', 'N/A')}")
                print(f"   Downloads: {ds.get('downloads', 'N/A')}")
                print(f"   Tags: {ds.get('tags', [])}")
                print(f"   Last Modified: {ds.get('last_modified', 'N/A')}")
                print("---")
        else:
            print("No datasets found matching the generated keywords.")

    except ValueError as ve: # Catch potential ValueError from llm_interaction setup
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 