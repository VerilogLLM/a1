from datasets import concatenate_datasets, Dataset

# Keep the fixed or optimized reasoning trace only
def combine_datasets(datasets_dict):
    """
    Combine multiple Hugging Face Datasets into a single Dataset with standardized columns.
    
    The output Dataset will have the following columns:
      - question
      - solution
      - reasoning_trace
      - data_source
      - category
      - metadata
    
    For each row:
      - 'question' is taken from the "question" column if present, else from "problem".
      - 'solution' is taken from the "solution" column if present, else from "answer".
      - All reasoning_* columns are set to empty strings (to be filled later).
      - 'data_source' is set to the dataset key.
      - 'category' is derived from "question-type" (and "subcategory" if available).
      - 'metadata' contains the remaining columns (excluding those already used) as a string.
    
    Args:
        datasets_dict (dict): A dictionary mapping dataset names to Hugging Face Dataset objects.
    
    Returns:
        Dataset: The combined and standardized Dataset with only the desired columns.
    """
    transformed_datasets = []
    # For each dataset, add a "data_source" column and transform each row,
    # while removing the original columns.
    for ds_key, ds in datasets_dict.items():
        transformed_ds = ds.map(
            lambda row, ds_key=ds_key: {
                "question": row.get("question", row.get("problem", "")),
                "solution": row.get("solution", row.get("answer", "")),
                "reasoning_plan": "",
                "reasoning_trace": "",
                "data_source": ds_key,
                "category": (row["question-type"] + (" - " + row["subcategory"] if "subcategory" in row else "")) 
                            if "question-type" in row else "",
                # "metadata": str({k: v for k, v in row.items() if k not in {"question", "problem", "solution", "answer", "question-type", "subcategory"}})
            },
            remove_columns=ds.column_names  # Remove original columns
        )
        transformed_datasets.append(transformed_ds)
    
    # Concatenate all transformed datasets into one.
    combined_dataset = concatenate_datasets(transformed_datasets)
    return combined_dataset

def combine_datasets_expand(datasets_dict, category):
    """
    Combine multiple Hugging Face Datasets into a single Dataset with standardized columns.
    
    Returns:
        Dataset: The combined and standardized Dataset with only the desired columns.
    """
    transformed_datasets = []
    # For each dataset, add a "data_source" column and transform each row,
    # while removing the original columns.
    for ds_key, ds in datasets_dict.items():
        transformed_ds = ds.map(
            lambda row, ds_key=ds_key: {
                "question": row.get("question", row.get("problem", "")),
                "solution": row.get("solution", row.get("answer", "")),
                "reasoning_plan": "",
                "reasoning_trace": "",
                "reasoning_step_score": "",
                "fixed_plan": "",
                "fixed_reasoning": "",
                "optimized_plan": "",
                "optimized_reasoning": "",
                "solution_hint_plan": "",
                "solution_hint_reasoning": "",
                "data_source": ds_key,
                "category": (row["question-type"] + (" - " + row["subcategory"] if "subcategory" in row else "")) 
                            if "question-type" in row else category,
                # "metadata": str({k: v for k, v in row.items() if k not in {"question", "problem", "solution", "answer", "question-type", "subcategory"}})
            },
            remove_columns=ds.column_names  # Remove original columns
        )
        transformed_datasets.append(transformed_ds)
    
    # Concatenate all transformed datasets into one.
    combined_dataset = concatenate_datasets(transformed_datasets)
    return combined_dataset

# Example usage:
if __name__ == "__main__":
    # Create dummy datasets that simulate the input.
    ds1 = Dataset.from_dict({
        "source": ["S1"] * 3,
        "problem": ["What is 2+2?", "What is the derivative of x^2?", "Solve x+3=5."],
        "solution": ["4", "2x", "2"],
        "messages": ["msg1", "msg2", "msg3"],
    })

    ds2 = Dataset.from_dict({
        "source": ["S2"] * 2,
        "problem": ["Compute the integral of 1/x.", "What is the limit of (1+1/n)^n?"],
        "solution": ["ln|x|", "e"],
        "messages": ["msgA", "msgB"],
        "gpt_difficulty": [3, 4],
        "gpt_difficulty_parsed": ["medium", "medium"],
    })

    ds3 = Dataset.from_dict({
        "question": ["What is 5-3?", "Solve 10/2."],
        "answer": ["2", "5"],
    })

    ds4 = Dataset.from_dict({
        "question-type": ["Algebra", "Calculus"],
        "subcategory": ["Linear", "Differentiation"],
        "question": ["Solve for x: 2x=4", "Find the derivative of sin(x)"],
        "answer": ["2", "cos(x)"]
    })

    datasets_dict = {
        "AI-MO/NuminaMath-CoT": ds1,
        "NovaSky-AI/labeled_numina_difficulty_162K": ds2,
        "openai/GSM8K": ds3,
        "davidheineman/deepmind-math-large": ds4
    }

    combined_ds = combine_datasets(datasets_dict)
    print(combined_ds)