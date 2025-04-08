from huggingface_hub import list_datasets
from huggingface_hub.utils import HfHubHTTPError

def search_datasets(
    keywords: list[str], 
    hf_token: str | None = None, 
    max_keywords_to_search: int = 3, 
    fetch_limit_per_keyword: int = 30,
    final_result_limit: int = 50
) -> list[dict]:
    """
    Searches the Hugging Face Hub for datasets based on the top keywords.
    Performs separate searches for each of the top `max_keywords_to_search` keywords.
    Combines unique results and sorts them by downloads.

    Args:
        keywords: A list of search keywords generated by the LLM.
        hf_token: Optional Hugging Face Hub token.
        max_keywords_to_search: Number of top keywords to search individually.
        fetch_limit_per_keyword: Limit results fetched per individual keyword search.
        final_result_limit: Max number of unique results to return after combining.

    Returns:
        A list of dictionaries, each containing info about a found dataset (up to `final_result_limit`).
        Returns an empty list if no datasets are found or an error occurs.
    """
    if not keywords:
        print("No keywords provided for searching.")
        return []

    # Use the top N keywords for individual searches
    keywords_to_search = keywords[:max_keywords_to_search]
    print(f"--- Performing individual searches for top {len(keywords_to_search)} keywords: {keywords_to_search} ---")

    all_found_datasets = {} # Use dict to store unique datasets by ID

    for i, keyword in enumerate(keywords_to_search):
        print(f"--- Searching for keyword ({i+1}/{len(keywords_to_search)}): '{keyword}' ---")
        try:
            dataset_generator = list_datasets(
                search=keyword,
                sort="downloads",
                direction=-1,
                limit=fetch_limit_per_keyword, # Limit per keyword search
                token=hf_token, # Pass token dynamically
                full=True # Fetch full metadata including description, likes etc.
            )
            
            count = 0
            for ds_info in dataset_generator:
                if ds_info.id not in all_found_datasets:
                    all_found_datasets[ds_info.id] = ds_info
                    count += 1
            print(f"--- Found {count} new unique datasets for keyword '{keyword}' ---")

        except HfHubHTTPError as e:
            print(f"Error searching Hugging Face Hub with keyword '{keyword}': {e}")
            continue 
        except Exception as e:
             print(f"An unexpected error occurred during search for '{keyword}': {e}")
             continue

    # If no datasets were found across all searches
    if not all_found_datasets:
        print(f"--- No datasets found for any of the top keywords: {keywords_to_search} ---")
        return []

    # Sort combined unique datasets by downloads, descending
    sorted_datasets = sorted(all_found_datasets.values(), key=lambda ds: ds.downloads or 0, reverse=True)
    
    print(f"--- Found a total of {len(sorted_datasets)} unique datasets across all keywords. Returning top {min(len(sorted_datasets), final_result_limit)} (sorted by downloads) ---")

    # Return the top N unique datasets
    return [
        {
            "id": ds.id,
            "author": ds.author,
            "tags": ds.tags,
            "downloads": ds.downloads,
            "likes": getattr(ds, 'likes', 0), # Add likes, use getattr for safety
            "description": getattr(ds, 'description', ""), # Add description
            "last_modified": ds.last_modified,
        }
        for ds in sorted_datasets[:final_result_limit]
    ]

# Example usage
if __name__ == '__main__':
    test_keywords = ["student academic performance", "education", "student achievement", "grades", "prediction"]
    print(f"Original Keywords: {test_keywords}")
    found_datasets = search_datasets(test_keywords)
    if found_datasets:
        # Display only the first 10 for command-line testing convenience
        print(f"\n--- Example Search Results (Top {min(10, len(found_datasets))}) ---") 
        for i, ds in enumerate(found_datasets[:10]):
            print(f"{i+1}. ID: {ds['id']}, Downloads: {ds.get('downloads', 'N/A')}, Likes: {ds.get('likes', 'N/A')}, Author: {ds.get('author', 'N/A')}, Desc: {ds.get('description', '')[:50]}...") # Also print likes and description snippet
    else:
        print("Example search returned no results.")