import pandas as pd
import re
import sys
import llm
import time
import os

def extract_answer(text):
    """
    Extracts the final numeric answer from a solution string.
    First, it looks for a LaTeX boxed answer (e.g., \\boxed{31}).
    If not found, it returns the last integer found in the text.
    """
    if text is None or text == "nan":
        return None
    if not isinstance(text, str):
        text = str(text)

    # Try to find a boxed answer in LaTeX format
    boxed = re.search(r'\$\\boxed\{(.*?)\}\$', text)
    if boxed:
        return boxed.group(1)

    # Check for True/False
    stripped = text.strip()
    if stripped in ["True", "False"]:
        return stripped == "True"

    # Try \boxed{number}
    # boxed_num = re.search(r'\\boxed\{(\d+)\}', text)
    boxed_num = re.search(r'\\boxed\{([\d.]+)\}', text)
    if boxed_num:
        return float(boxed_num.group(1)) if '.' in boxed_num.group(1) else int(boxed_num.group(1))

    # Won't fit to all cases.
    # Last numeric value (integer or float) in text
    m = re.findall(r"-?\d+\.?\d*", text)
    if m:
        return float(m[-1]) if '.' in m[-1] else int(m[-1])

    return None

def numeric_equal(a, b):
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False

    # If the solution is correct, write the template:
    # Correctness:{{YES}}
    # If the solution is incorrect, write the template:
    # Correctness:{{NO}}
def prompt(ground_truth, reasoning_text):
    return f"""
    You are a judge that evaluates the correctness of a solution.
    You will be given a solution and a ground truth solution.
    You will need to determine if the solution is correct.
    Answers are in the format of \\boxed{{}}.
    If answer includes an empty box {{}}, then backtrace the solution and find the answer.

    SOLUTION: {reasoning_text}
    GROUND TRUTH SOLUTION: {ground_truth}

    If they reach the same conclusion, print match in \\boxed{{match}}.
    """

def compare_solutions_llm(solution_text, reasoning_text):
    """
    Uses the LLM to compare the solution and reasoning texts.
    Returns a tuple (match_bool, critique_str).
    """
    system_prompt = "You are a judge that evaluates the correctness of a solution."
    # user_prompt = f"Solution: {solution_text}\nReasoning: {reasoning_text}\nIf they reach the same conclusion, answer 'yes', otherwise 'no'."
    user_prompt = prompt(solution_text, reasoning_text)

    response = llm.llm_ollama("gemma3:12b", system_prompt, user_prompt)
    print("LLM response:\n", response)
    # match = 'yes' in response.lower()
    match = '{match}' in response.lower()
    return match, response.strip()

def compare_solutions(solution_text, reasoning_text, equality_check=False):
    """
    Compares the numerical answer extracted from the full solution text and
    the reasoning text, falling back to LLM if needed.
    Returns (match_bool, critique_str).
    """
    ans_sol = extract_answer(solution_text)
    ans_rat = extract_answer(reasoning_text)

    # String comparison first in case of converting True to numerical 1.
    if str(ans_sol) == str(ans_rat):
        critique = f"String match: solution_answer={ans_sol}, reasoning_answer={ans_rat}"
        print("Result: The solutions match.\n")
        return True, critique

    # Numeric comparison
    if ans_sol == ans_rat or numeric_equal(ans_sol, ans_rat):
        critique = f"Numeric match: solution_answer={ans_sol}, reasoning_answer={ans_rat}"
        return True, critique

    if equality_check:
        return False, None

    # Fallback to LLM comparison
    return compare_solutions_llm(solution_text, reasoning_text)

def get_reasoning_text(row):
    """
    Chooses the reasoning text based on priority:
    1. solution_hint_reasoning if available,
    2. optimized_reasoning if available,
    3. fixed_reasoning if available,
    4. reasoning_trace if available,
    """
    for col in ['solution_hint_reasoning', 'optimized_reasoning', 'fixed_reasoning', 'reasoning_trace']:
        if isinstance(row.get(col), str) and row[col].strip():
            return row[col]
    return ''

def get_candidate_reasoning_plan(row):
    """
    Chooses the corresponding plan column based on which reasoning field was selected:
    - If solution_hint_reasoning, pick solution_hint_plan
    - If optimized_reasoning, pick optimized_plan
    - If fixed_reasoning, pick fixed_plan
    - If reasoning_trace, pick reasoning_plan
    """
    mapping = [
        ('solution_hint_reasoning', 'solution_hint_plan'),
        ('optimized_reasoning',    'optimized_plan'),
        ('fixed_reasoning',        'fixed_plan'),
        ('reasoning_trace',        'reasoning_plan'),
    ]
    for reasoning_col, plan_col in mapping:
        if isinstance(row.get(reasoning_col), str) and row[reasoning_col].strip():
            return row.get(plan_col, '')
    return ''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cluster_results.py <path_to_csv>")
        sys.exit(1)
    outdir = os.path.abspath("./output")

    start_time = time.time()
    df = pd.read_csv(sys.argv[1])

    control = 'solution'
    # compare = 'solution_hint_reasoning'
    # df['solution_candidate'] = df[compare].apply(lambda x: x if isinstance(x, str) else '')
    df['solution_candidate'] = df.apply(get_reasoning_text, axis=1)
    # Drop rows where solution_candidate is empty
    df = df[df['solution_candidate'] != ""]

    df['candidate_reasoning_plan']   = df.apply(get_candidate_reasoning_plan, axis=1)

    # Extract answers
    df['solution_answer'] = df[control].apply(extract_answer)
    df['reasoning_answer'] = df['solution_candidate'].apply(extract_answer)

    # Compare and collect critiques
    results = df.apply(
        lambda row: compare_solutions(row[control], row['solution_candidate']),
        axis=1, result_type='expand'
    )
    df['box_match'] = results[0]
    df['critique'] = results[1]

    # Split into matching and non-matching rows
    matching_df = df[df['box_match']]
    non_matching_df = df[~df['box_match']]

    # Combine both sets and select only required columns
    combined_df = pd.concat([matching_df, non_matching_df], ignore_index=True)
    output_df = combined_df[['solution', 'question', 'solution_candidate', 'candidate_reasoning_plan', 'critique']]

    # Write the combined dataframe to CSV and JSON
    output_df.to_csv(os.path.join(outdir, "combined_ds.csv"), index=False)
    output_df.to_json(os.path.join(outdir, "combined_ds.json"), orient="records", lines=True)

    # Write output files
    matching_df.to_csv(os.path.join(outdir, "solution_matching_rows.csv"), index=False)
    matching_df.to_json(os.path.join(outdir, "solution_matching_rows.json"), orient="records", lines=True)

    non_matching_df.to_csv(os.path.join(outdir, "solution_non_matching_rows.csv"), index=False)
    non_matching_df.to_json(os.path.join(outdir, "solution_non_matching_rows.json"), orient="records", lines=True)

    print("Files written:")
    print(f" - {os.path.join(outdir, 'solution_matching_rows.csv')}")
    print(f" - {os.path.join(outdir, 'solution_matching_rows.json')}")
    print(f" - {os.path.join(outdir, 'solution_non_matching_rows.csv')}")
    print(f" - {os.path.join(outdir, 'solution_non_matching_rows.json')}")
    print(f" - {os.path.join(outdir, 'combined_ds.csv')}")
    print(f" - {os.path.join(outdir, 'combined_ds.json')}")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
