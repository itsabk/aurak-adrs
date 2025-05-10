from typing import List, Optional, TypedDict

# Predefined options for dropdowns
PREDEFINED_DOMAINS = [
    "All Domains",
    "NLP",
    "Computer Vision",
    "Healthcare",
    "Finance",
    "Social Science",
    "Audio",
    "Robotics",
    "Time Series",
    "Biology",
    "Education"
]

PREDEFINED_TASKS = [
    "All Tasks",
    "Classification",
    "Regression",
    "Object Detection",
    "Text Generation",
    "Translation",
    "Question Answering",
    "Named Entity Recognition",
    "Sentiment Analysis",
    "Image Classification",
    "Speech Recognition"
]

DATA_SIZE_OPTIONS = [
    "Any Size",
    "Small (<1k)",
    "Medium (1k-100k)",
    "Large (100k-1M)",
    "Very Large (>1M)"
]

PREDEFINED_LICENSES = [
    "All Licenses",
    "MIT",
    "Apache-2.0",
    "CC-BY",
    "CC-BY-SA",
    "CC-BY-NC",
    "GPL-3.0",
    "BSD"
]

QUALITY_CRITERIA = [
    "All Quality Levels",
    "Documentation Required",
    "Data Validation",
    "Citation Information",
    "Test Split Available",
    "Validation Split Available",
    "Data Cleaning Applied",
    "Quality Metrics Available"
]

PREDEFINED_LANGUAGES = [
    "All Languages",
    "English",
    "Spanish",
    "French",
    "German",
    "Chinese",
    "Arabic",
    "Hindi",
    "Japanese",
    "Russian",
    "Portuguese"
]

TIME_RANGES = [
    "Any Time",
    "Last 6 months",
    "Last year",
    "Last 2 years",
    "Last 5 years"
]

class AdvancedSearchOptions(TypedDict, total=False):
    """
    Defines the structure for advanced search options.
    All fields are optional and default to None/empty list if not specified.
    """
    domains: List[str]              # Selected domains (including custom)
    tasks: List[str]               # Selected task types (including custom)
    data_sizes: List[str]          # Selected data size ranges
    licenses: List[str]            # Selected license types (including custom)
    quality_criteria: List[str]    # Selected quality requirements (including custom)
    languages: List[str]           # Selected languages (including custom)
    time_range: str               # Selected time range
    custom_domains: List[str]      # User-added custom domains
    custom_tasks: List[str]       # User-added custom tasks
    custom_licenses: List[str]    # User-added custom licenses
    custom_quality: List[str]     # User-added custom quality criteria
    custom_languages: List[str]   # User-added custom languages
    custom_time_range: Optional[str] # User-specified custom time range

def get_default_options() -> AdvancedSearchOptions:
    """
    Returns the default advanced search options.
    """
    return {
        "domains": ["All Domains"],
        "tasks": ["All Tasks"],
        "data_sizes": ["Any Size"],
        "licenses": ["All Licenses"],
        "quality_criteria": ["All Quality Levels"],
        "languages": ["All Languages"],
        "time_range": "Any Time",
        "custom_domains": [],
        "custom_tasks": [],
        "custom_licenses": [],
        "custom_quality": [],
        "custom_languages": [],
        "custom_time_range": None
    } 