import os
import time
import argparse
import logging
import tqdm
from datasets import load_dataset

# Local imports
import deduplicate
import combine
import engine

# Global logger configuration
def configure_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Dataset loading helpers
def load_dataset_safe(name, hf_id=None, split="train", **kwargs):
    try:
        ds = load_dataset(name, split=split, **kwargs) if hf_id else load_dataset(name, split=split, **kwargs)
        print(f"Loaded {name} [{split}]")
        return ds
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None

def load_dataset_gsm(name, subset, split="train", **kwargs):
    """Loads GSM8K (or similar) dataset using a subset identifier."""
    return load_dataset(name, subset, split=split, **kwargs)

def load_trainset(selection, shuffle=False, num_rows=None):
    math_datasets = {}
    for key in tqdm.tqdm(selection, desc="Loading train datasets"):
        if key == "openai/GSM8K":
            ds = load_dataset_gsm(key, "main", split="train")
        else:
            ds = load_dataset_safe(key, split="train")

        if ds:
            if shuffle:
                ds = ds.shuffle(seed=42)
                print(f"Shuffled dataset: {key}")
            if num_rows:
                ds = ds.select(range(min(num_rows, len(ds))))
                print(f"Selected first {num_rows} rows from dataset: {key}")
            math_datasets[key] = ds
    return math_datasets


def load_ds_questions(dataset_list):
    questions = []
    for name in tqdm.tqdm(dataset_list, desc="Loading eval datasets"):
        if name == "openai/GSM8K":
            ds = load_dataset_gsm(name, "main", split="test")
            data = ds
        else:
            ds = load_dataset(name)
            split = 'test' if 'test' in ds.keys() else list(ds.keys())[0]
            data = ds[split]

        col = 'question' if 'question' in data.column_names else 'problem' if 'problem' in data.column_names else None
        if col is not None:
            questions.extend(data[col])
    return questions


def main(train_selection, test_selection):
    parser = argparse.ArgumentParser(description="LLM Reasoning Pipeline")
    parser.add_argument('-l', '--log-file', type=str, default=None)
    parser.add_argument('-n', '--num-rows', type=int, default=100)
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-m', '--model', type=str, default='anthropic')
    parser.add_argument('-mode', '--mode', type=int, default=0)
    parser.add_argument('-csv', '--csv-file', type=str, help="Path to an existing CSV file")
    args = parser.parse_args()

    # Setup logger
    global logger
    logger = configure_logger(args.log_file)
    start = time.time()

    if args.mode == 2:
        if not args.csv_file:
            print("Error: --csv-file argument is required when mode is 2.")
            exit(1)
    if args.csv_file:
        # Load existing CSV
        import pandas as pd
        from datasets import Dataset
        pd_csv = pd.read_csv(args.csv_file)
        print(f"Loaded existing CSV file: {args.csv_file}")
        combined = Dataset.from_pandas(pd_csv)
        if args.mode == 2:
            # Drop unnecessary columns
            columns_to_keep = ['question', 'solution', 'solution_candidate', 'candidate_reasoning_plan', 'critique']
            combined = combined.remove_columns([col for col in combined.column_names if col not in columns_to_keep])
    else:
        # Load datasets
        train_sets = load_trainset(train_selection, shuffle=True, num_rows=args.num_rows)
        eval_qs = load_ds_questions(test_selection)
        print(f"Number of training datasets: {len(train_sets)}")
        print(f"Number of evaluation questions: {len(eval_qs)}")

        # Deduplication
        deduped = {}
        for k, ds in train_sets.items():
            res = deduplicate.deduplicate_train_from_test_ngram_ds_list(
                ds, eval_qs, n=3, threshold=0.8, cosine=False
            )
            if res:
                deduped[k] = res
        print(f"Number of deduplicated datasets: {len(deduped)}")

        # Combine
        combined = combine.combine_datasets_expand(deduped, category="math")

    outdir = os.path.abspath("./output")
    os.makedirs(outdir, exist_ok=True)
    combined.to_json(
        os.path.join(outdir, "train_ds_math.json"), orient='records', lines=True
    )
    combined.to_csv(
        os.path.join(outdir, "train_ds_math.csv"), index=False
    )

    # Process with engine
    model_type = args.model
    anthropic_keys = [
        os.getenv("ANTHROPIC_API_KEY"), os.getenv("ANTHROPIC_API_KEY1"),
        os.getenv("ANTHROPIC_API_KEY2")
    ]
    gemini_keys = [
        os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY1"),
        os.getenv("GEMINI_API_KEY2")
    ]
    verif_keys = [
        os.getenv("OPENAI_API_KEY_VERIF"), os.getenv("OPENAI_API_KEY_VERIF1")
    ]
    api_keys = gemini_keys if model_type.lower() == 'gemini' else anthropic_keys

    if args.parallel:
        updated_ds, tokens = engine.process_api_reasoning_parallel(
            combined, api_keys, verif_keys, model_type, outdir, args.mode
        )
    else:
        updated_ds, tokens = engine.process_api_reasoning(
            combined, model_type, outdir, args.mode
        )

    # Print summary
    engine.print_token_summary(tokens, model_type)

    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.2f}s")


if __name__ == '__main__':
    TRAIN_SELECTION = [
        "AI-MO/NuminaMath-1.5", 
        "openai/GSM8K",
        "davidheineman/deepmind-math-large",
        "active-reasoning/AIME"
    ]
    TEST_SELECTION = [
        "AI-MO/NuminaMath-CoT",
    ]
    main(TRAIN_SELECTION, TEST_SELECTION)
