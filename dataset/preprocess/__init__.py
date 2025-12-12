from .mmk12 import preprocess_mmk12


def get_preprocess_func(dataset_path: str):
    # Match dataset path patterns to their preprocessing functions
    if "MMK12" in dataset_path or "mmk12" in dataset_path.lower():
        return preprocess_mmk12
    
    # Default: no preprocessing
    return lambda x: x
