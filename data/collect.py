from datasets import load_dataset

def load_dataset_safe(name, hf_id=None, split="train", **kwargs):
    """
    Tries to load a dataset using the given Hugging Face identifier with a specified split.
    If hf_id is provided, it loads that dataset; otherwise, it uses `name`.
    In case of error, it prints a message and returns None.
    """
    split = "train[1:10]"
    try:
        ds = load_dataset(hf_id, split=split, **kwargs) if hf_id else load_dataset(name, split=split, **kwargs)
        print(f"Loaded {name} successfully with split='{split}'.")
        return ds
    except Exception as e:
        print(f"Failed to load {name} with split='{split}': {e}")
        return None

def main():
    dataset_dict = {}

    # ==========================
    # Math Datasets
    # ==========================
    math_datasets = {}
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    # These datasets (AIME, AMC, Math Olympiad) might not be on HF Hub; placeholders are provided.
    math_datasets['AIME'] = None  # Custom loader required
    math_datasets['AMC'] = None   # Custom loader required
    math_datasets['Math_Olympiad'] = None  # Custom loader required

    # GSM8K is available as openai/gsm8k.
    math_datasets['GSM8K'] = load_dataset_safe("GSM8K", hf_id="openai/gsm8k", split="train")
    
    # "MATH" is a competition-level math dataset.
    math_datasets['MATH'] = None  # Placeholder: Implement custom loader if available

    # DeepMind Mathematics dataset is available on Hugging Face.
    math_datasets['DeepMind_Mathematics'] = load_dataset_safe("DeepMind_Mathematics", hf_id="deepmind/math_dataset", split="train")

    dataset_dict['Math'] = math_datasets

    # ==========================
    # Science Datasets
    # ==========================
    science_datasets = {}
    # GPQA might need a custom loader if not available on HF Hub.
    science_datasets['GPQA'] = None  # Placeholder: Implement custom loader if available

    science_datasets['SciQ'] = load_dataset_safe("SciQ", hf_id="sciq", split="train")
    science_datasets['ARC'] = load_dataset_safe("ARC", hf_id="arc", split="train")
    science_datasets['OpenBookQA'] = load_dataset_safe("OpenBookQA", hf_id="openbookqa", split="train")
    science_datasets['QASC'] = load_dataset_safe("QASC", hf_id="qasc", split="train")
    science_datasets['PubMedQA'] = load_dataset_safe("PubMedQA", hf_id="pubmed_qa", split="train")

    dataset_dict['Science'] = science_datasets

    # ==========================
    # Common Sense Datasets
    # ==========================
    commonsense_datasets = {}
    commonsense_datasets['CommonsenseQA'] = load_dataset_safe("CommonsenseQA", hf_id="commonsense_qa", split="train")
    commonsense_datasets['SWAG'] = load_dataset_safe("SWAG", hf_id="swag", split="train")
    commonsense_datasets['HellaSwag'] = load_dataset_safe("HellaSwag", hf_id="hellaswag", split="train")
    # Using Winogrande as a proxy for the Winograd Schema Challenge.
    commonsense_datasets['Winograd'] = load_dataset_safe("Winogrande", hf_id="winogrande", split="train")
    commonsense_datasets['CycIC'] = None  # Placeholder: Custom loader needed if available
    commonsense_datasets['PhysicalIQA'] = load_dataset_safe("PhysicalIQA", hf_id="physicaliqa", split="train")

    dataset_dict['CommonSense'] = commonsense_datasets

    # ==========================
    # Coding Datasets
    # ==========================
    coding_datasets = {}
    coding_datasets['SWE_bench'] = None  # Placeholder: Implement custom loader if available
    coding_datasets['CodeSearchNet'] = load_dataset_safe("CodeSearchNet", hf_id="code_search_net", split="train")
    coding_datasets['HumanEval'] = load_dataset_safe("HumanEval", hf_id="human_eval", split="train")
    coding_datasets['MBPP'] = load_dataset_safe("MBPP", hf_id="mbpp", split="train")
    coding_datasets['APPS'] = load_dataset_safe("APPS", hf_id="apps", split="train")
    coding_datasets['CodeXGLUE'] = load_dataset_safe("CodeXGLUE", hf_id="codex_glue", split="train")

    dataset_dict['Coding'] = coding_datasets

    # ==========================
    # Additional Notable Datasets
    # ==========================
    additional_datasets = {}
    additional_datasets['MMLU'] = load_dataset_safe("MMLU", hf_id="mmlu", split="train")
    additional_datasets['BIG_bench'] = load_dataset_safe("BIG-bench", hf_id="bigbench", split="train")
    additional_datasets['Hendrycks_Math'] = None  # Placeholder: DMCA or custom loader may be needed
    additional_datasets['SciFact'] = load_dataset_safe("SciFact", hf_id="scifact", split="train")

    dataset_dict['Additional'] = additional_datasets

    print("Dataset loading complete.")
    return dataset_dict

if __name__ == "__main__":
    datasets_loaded = main()
    print(datasets_loaded)