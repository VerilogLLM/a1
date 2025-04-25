import math
from collections import Counter
from fuzzywuzzy import fuzz
import tqdm
from datasets import Dataset

### Jaccard Similarity for N-Grams
def get_ngrams(text, n=3):
    """
    Generate a set of word-level n-grams from the input text.
    
    Args:
        text (str): The input text.
        n (int): The n-gram size.
        
    Returns:
        set: A set of n-gram tuples.
    """
    tokens = text.lower().split()
    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.add(ngram)
    return ngrams

def jaccard_similarity(ngrams1, ngrams2):
    """
    Compute the Jaccard similarity between two sets of n-grams.
    
    Args:
        ngrams1 (set): First set of n-grams.
        ngrams2 (set): Second set of n-grams.
        
    Returns:
        float: Jaccard similarity score.
    """
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)

### Cosine Similarity for N-Grams
def get_ngram_counter(text, n=3):
    """
    Generate a frequency counter of word-level n-grams from the input text.

    Args:
        text (str): The input text.
        n (int): The n-gram size (default is 3).

    Returns:
        Counter: A collections.Counter object with n-gram frequencies.
    """
    tokens = text.lower().split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)

def cosine_similarity(counter1, counter2):
    """
    Compute the cosine similarity between two n-gram frequency counters.

    Args:
        counter1 (Counter): The n-gram frequency counter for document 1.
        counter2 (Counter): The n-gram frequency counter for document 2.

    Returns:
        float: Cosine similarity score between the two counters.
    """
    dot = sum(counter1[ng] * counter2.get(ng, 0) for ng in counter1)
    norm1 = math.sqrt(sum(val * val for val in counter1.values()))
    norm2 = math.sqrt(sum(val * val for val in counter2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

##################################################
# List-based deduplication functions (for reference)
##################################################

def deduplicate_train_from_test_ngram(train_dataset, test_dataset, n=3, threshold=0.8, cosine=False):
    """
    Remove entries from the training dataset (list of strings) that are highly similar to any test document,
    based on n-gram Jaccard similarity (or cosine similarity if cosine=True).

    Args:
        train_dataset (list of str): List of training documents.
        test_dataset (list of str): List of test documents.
        n (int): The n-gram size for similarity computation.
        threshold (float): Similarity threshold above which a training document is considered a duplicate.
        cosine (bool): Whether to use cosine similarity instead of Jaccard similarity.

    Returns:
        list of str: Deduplicated training dataset.
    """
    if cosine:
        test_ngram_counters = [get_ngram_counter(doc, n) for doc in test_dataset]
    else:
        test_ngrams_list = [get_ngrams(doc, n) for doc in test_dataset]
    
    deduplicated_train = []
    seen_train = set()

    for doc in tqdm.tqdm(train_dataset, desc="Deduplicating training data (ngram)"):
        normalized_doc = ' '.join(doc.lower().split())
        if normalized_doc in seen_train:
            continue
        seen_train.add(normalized_doc)

        if cosine:
            train_counter = get_ngram_counter(doc, n)
        else:
            train_ngrams = get_ngrams(doc, n)
        
        remove_flag = False
        if cosine:
            for test_counter in test_ngram_counters:
                similarity = cosine_similarity(train_counter, test_counter)
                if similarity >= threshold:
                    remove_flag = True
                    break
        else:
            for test_ngrams in test_ngrams_list:
                similarity = jaccard_similarity(train_ngrams, test_ngrams)
                if similarity >= threshold:
                    remove_flag = True
                    break
  
        if not remove_flag:
            deduplicated_train.append(doc)
            
    return deduplicated_train

def deduplicate_train_from_test_fuzzy(train_dataset, test_dataset, threshold=90):
    """
    Remove entries from the training dataset (list of strings) that are too similar to any test document,
    based on fuzzy string matching.

    Args:
        train_dataset (list of str): List of training documents.
        test_dataset (list of str): List of test documents.
        threshold (int): Fuzzy similarity threshold (0 to 100).

    Returns:
        list of str: Deduplicated training dataset.
    """
    normalized_test = [' '.join(doc.lower().split()) for doc in test_dataset]
    
    deduplicated_train = []
    seen_train = set()

    for doc in tqdm.tqdm(train_dataset, desc="Deduplicating training data (fuzzy)"):
        norm_doc = ' '.join(doc.lower().split())
        if norm_doc in seen_train:
            continue
        seen_train.add(norm_doc)
        
        remove_flag = False
        for test_doc in normalized_test:
            similarity = fuzz.ratio(norm_doc, test_doc)
            if similarity >= threshold:
                remove_flag = True
                break
        
        if not remove_flag:
            deduplicated_train.append(doc)
    
    return deduplicated_train

##################################################
# Dataset-based deduplication functions (both inputs as Dataset objects)
##################################################

def deduplicate_train_from_test_ngram_ds(train_dataset, test_dataset, n=3, threshold=0.8, cosine=False, train_column=None, test_column=None):
    """
    Deduplicate a training Dataset based on similarity with a test Dataset.
    The text is extracted from a specified column (defaulting to "question" if available, otherwise "problem").
    Returns a new Dataset with rows removed that are highly similar to any test document.

    Args:
        train_dataset (Dataset): Hugging Face Dataset for training data.
        test_dataset (Dataset): Hugging Face Dataset for test data.
        n (int): The n-gram size for similarity computation.
        threshold (float): Similarity threshold above which a training document is considered a duplicate.
        cosine (bool): Whether to use cosine similarity instead of Jaccard similarity.
        train_column (str, optional): Column name for text in training data.
        test_column (str, optional): Column name for text in test data.

    Returns:
        Dataset: A new Dataset containing only the deduplicated training rows.
    """
    # Determine which column to use for training data
    if train_column is None:
        if "question" in train_dataset.column_names:
            train_column = "question"
        elif "problem" in train_dataset.column_names:
            train_column = "problem"
        else:
            raise ValueError("No suitable text column found in the training dataset.")

    # Determine which column to use for test data
    if test_column is None:
        if "question" in test_dataset.column_names:
            test_column = "question"
        elif "problem" in test_dataset.column_names:
            test_column = "problem"
        else:
            raise ValueError("No suitable text column found in the test dataset.")

    train_texts = train_dataset[train_column]
    test_texts = test_dataset[test_column]

    kept_indices = []
    seen_train = set()

    if cosine:
        test_ngram_counters = [get_ngram_counter(doc, n) for doc in test_texts]
    else:
        test_ngrams_list = [get_ngrams(doc, n) for doc in test_texts]

    for idx, doc in tqdm.tqdm(enumerate(train_texts), total=len(train_texts), desc="Deduplicating dataset (ngram)"):
        normalized_doc = ' '.join(doc.lower().split())
        if normalized_doc in seen_train:
            continue
        seen_train.add(normalized_doc)

        remove_flag = False
        if cosine:
            train_counter = get_ngram_counter(doc, n)
            for test_counter in test_ngram_counters:
                similarity = cosine_similarity(train_counter, test_counter)
                if similarity >= threshold:
                    remove_flag = True
                    break
        else:
            train_ngrams = get_ngrams(doc, n)
            for test_ngrams in test_ngrams_list:
                similarity = jaccard_similarity(train_ngrams, test_ngrams)
                if similarity >= threshold:
                    remove_flag = True
                    break

        if not remove_flag:
            kept_indices.append(idx)

    return train_dataset.select(kept_indices)

def deduplicate_train_from_test_fuzzy_ds(train_dataset, test_dataset, threshold=90, train_column=None, test_column=None):
    """
    Deduplicate a training Dataset using fuzzy string matching based on similarity with a test Dataset.
    The text is extracted from a specified column (defaulting to "question" if available, otherwise "problem").
    Returns a new Dataset with rows removed that are too similar to any test document.

    Args:
        train_dataset (Dataset): Hugging Face Dataset for training data.
        test_dataset (Dataset): Hugging Face Dataset for test data.
        threshold (int): Fuzzy similarity threshold (0 to 100).
        train_column (str, optional): Column name for text in training data.
        test_column (str, optional): Column name for text in test data.

    Returns:
        Dataset: A new Dataset containing only the deduplicated training rows.
    """
    if train_column is None:
        if "question" in train_dataset.column_names:
            train_column = "question"
        elif "problem" in train_dataset.column_names:
            train_column = "problem"
        else:
            raise ValueError("No suitable text column found in the training dataset.")

    if test_column is None:
        if "question" in test_dataset.column_names:
            test_column = "question"
        elif "problem" in test_dataset.column_names:
            test_column = "problem"
        else:
            raise ValueError("No suitable text column found in the test dataset.")

    train_texts = train_dataset[train_column]
    test_texts = test_dataset[test_column]

    normalized_test = [' '.join(doc.lower().split()) for doc in test_texts]
    kept_indices = []
    seen_train = set()

    for idx, doc in tqdm.tqdm(enumerate(train_texts), total=len(train_texts), desc="Deduplicating dataset (fuzzy)"):
        norm_doc = ' '.join(doc.lower().split())
        if norm_doc in seen_train:
            continue
        seen_train.add(norm_doc)
        remove_flag = False
        for test_doc in normalized_test:
            similarity = fuzz.ratio(norm_doc, test_doc)
            if similarity >= threshold:
                remove_flag = True
                break
        if not remove_flag:
            kept_indices.append(idx)

    return train_dataset.select(kept_indices)

##################################################
# New functions: Dataset-based deduplication with test data as list of strings
##################################################

def deduplicate_train_from_test_ngram_ds_list(train_dataset, test_questions, n=3, threshold=0.8, cosine=False, train_column=None):
    """
    Deduplicate a training Dataset based on similarity with a list of test question strings.
    The text is extracted from a specified column in the training dataset (defaulting to "question" if available, otherwise "problem").
    Returns a new Dataset with rows removed that are highly similar to any test question.

    Args:
        train_dataset (Dataset): Hugging Face Dataset for training data.
        test_questions (list of str): List of test question strings.
        n (int): The n-gram size for similarity computation.
        threshold (float): Similarity threshold above which a training document is considered a duplicate.
        cosine (bool): Whether to use cosine similarity instead of Jaccard similarity.
        train_column (str, optional): Column name for text in training data.

    Returns:
        Dataset: A new Dataset containing only the deduplicated training rows.
    """
    if train_column is None:
        if "question" in train_dataset.column_names:
            train_column = "question"
        elif "problem" in train_dataset.column_names:
            train_column = "problem"
        else:
            raise ValueError("No suitable text column found in the training dataset.")
    
    train_texts = train_dataset[train_column]
    kept_indices = []
    seen_train = set()

    if cosine:
        test_ngram_counters = [get_ngram_counter(doc, n) for doc in test_questions]
    else:
        test_ngrams_list = [get_ngrams(doc, n) for doc in test_questions]

    for idx, doc in tqdm.tqdm(enumerate(train_texts), total=len(train_texts), desc="Deduplicating dataset (ngram, test list)"):
        normalized_doc = ' '.join(doc.lower().split())
        if normalized_doc in seen_train:
            continue
        seen_train.add(normalized_doc)

        remove_flag = False
        if cosine:
            train_counter = get_ngram_counter(doc, n)
            for test_counter in test_ngram_counters:
                similarity = cosine_similarity(train_counter, test_counter)
                if similarity >= threshold:
                    remove_flag = True
                    break
        else:
            train_ngrams = get_ngrams(doc, n)
            for test_ngrams in test_ngrams_list:
                similarity = jaccard_similarity(train_ngrams, test_ngrams)
                if similarity >= threshold:
                    remove_flag = True
                    break

        if not remove_flag:
            kept_indices.append(idx)

    return train_dataset.select(kept_indices)

def deduplicate_train_from_test_fuzzy_ds_list(train_dataset, test_questions, threshold=90, train_column=None):
    """
    Deduplicate a training Dataset using fuzzy string matching based on similarity with a list of test question strings.
    The text is extracted from a specified column in the training dataset (defaulting to "question" if available, otherwise "problem").
    Returns a new Dataset with rows removed that are too similar to any test question.

    Args:
        train_dataset (Dataset): Hugging Face Dataset for training data.
        test_questions (list of str): List of test question strings.
        threshold (int): Fuzzy similarity threshold (0 to 100).
        train_column (str, optional): Column name for text in training data.

    Returns:
        Dataset: A new Dataset containing only the deduplicated training rows.
    """
    if train_column is None:
        if "question" in train_dataset.column_names:
            train_column = "question"
        elif "problem" in train_dataset.column_names:
            train_column = "problem"
        else:
            raise ValueError("No suitable text column found in the training dataset.")
    
    train_texts = train_dataset[train_column]
    normalized_test = [' '.join(doc.lower().split()) for doc in test_questions]
    kept_indices = []
    seen_train = set()

    for idx, doc in tqdm.tqdm(enumerate(train_texts), total=len(train_texts), desc="Deduplicating dataset (fuzzy, test list)"):
        norm_doc = ' '.join(doc.lower().split())
        if norm_doc in seen_train:
            continue
        seen_train.add(norm_doc)
        remove_flag = False
        for test_doc in normalized_test:
            similarity = fuzz.ratio(norm_doc, test_doc)
            if similarity >= threshold:
                remove_flag = True
                break
        if not remove_flag:
            kept_indices.append(idx)

    return train_dataset.select(kept_indices)

########################################
# Example usage:
########################################
if __name__ == "__main__":
    # Create dummy datasets for demonstration.
    train_ds = Dataset.from_dict({
        "question": [
            "This is a sample training document that has some unique content.",
            "Another training document with different content.",
            "Let x(n) be a sequence of real naumbers. Derivative of x(n) is given by x'(n).",
            "Let x(n) be a sequence of real numbers.",
            "This document is nearly identical to a test set document.",
            "This document is nearly identical to a test set document."
        ],
        "source": ["A", "B", "C", "D", "E", "F"]
    })

    test_ds = Dataset.from_dict({
        "question": [
            "This is a sample training document that has some unique content.",
            "Let x(n) be a sequence of real numbers. Derivative of x(n) is given by x'(n).",
            "An entirely different test document."
        ]
    })

    # Using the Dataset-based functions (both inputs as Datasets)
    deduped_train_ds_ngram = deduplicate_train_from_test_ngram_ds(train_ds, test_ds, n=3, threshold=0.8, cosine=False)
    print("Deduplicated Training Dataset (ngram-based, both as Datasets):")
    for row in deduped_train_ds_ngram:
        print(row)

    deduped_train_ds_fuzzy = deduplicate_train_from_test_fuzzy_ds(train_ds, test_ds, threshold=90)
    print("\nDeduplicated Training Dataset (fuzzy-based, both as Datasets):")
    for row in deduped_train_ds_fuzzy:
        print(row)

    # Using the new functions where test input is a list of question strings.
    test_questions_list = [
        "This is a sample training document that has some unique content.",
        "Let x(n) be a sequence of real numbers. Derivative of x(n) is given by x'(n).",
        "An entirely different test document."
    ]
    deduped_train_ds_ngram_list = deduplicate_train_from_test_ngram_ds_list(train_ds, test_questions_list, n=3, threshold=0.8, cosine=False)
    print("\nDeduplicated Training Dataset (ngram-based, test as list):")
    for row in deduped_train_ds_ngram_list:
        print(row)

    deduped_train_ds_fuzzy_list = deduplicate_train_from_test_fuzzy_ds_list(train_ds, test_questions_list, threshold=90)
    print("\nDeduplicated Training Dataset (fuzzy-based, test as list):")
    for row in deduped_train_ds_fuzzy_list:
        print(row)