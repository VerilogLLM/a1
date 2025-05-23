import os
import pandas as pd
import tempfile
import shutil
import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor

# Import local modules
import llm
import prompt
import estimate_cost
import cluster_results
from extraction_helpers import extract_reasoning_components, extract_verification_components, extract_solution_hint_components

# Global logger variable
logger = logging.getLogger(__name__)

# =============================================================================
# LLM Selection Helper
# =============================================================================
def call_llm(model_type, api_key, system_prompt, question):
    """
    Calls the appropriate LLM API based on the model_type.
    Returns (response, token_data) where token_data is a dummy for models that do not return token usage.
    """
    if model_type is None or model_type.lower() == "anthropic":
        # Use anthropic API with "sonnet-3.7"
        # Note: we pass the api_key since anthropic requires it.
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        return llm.llm_anthropic("claude-3-7-sonnet-20250219", system_prompt, question, api_key)
    elif model_type.lower() == "gemini":
        # Use gemini API with "gemini" (api key is read from env in llm_gemini)
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        # model="gemini-2.5-pro-preview-03-25",
        return llm.llm_gemini("gemini-2.0-flash", system_prompt, question, api_key)
    elif model_type.lower() == "o4-mini":
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        # return llm.llm_openai("o3-mini", system_prompt, question, api_key)
        return llm.llm_openai("o4-mini", system_prompt, question, api_key)
    elif model_type.lower() == "qwq":
        # Use vllm API with "grok qwq"
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        response = llm.llm_vllm("grok-qwq", system_prompt, question)
        # Create a dummy token_data object with zero counts
        DummyTokenData = type("DummyTokenData", (), {}) 
        dummy = DummyTokenData()
        dummy.input_tokens = 0
        dummy.output_tokens = 0
        return response, dummy
    else:
        raise ValueError("Unknown model type: " + model_type)

def token_counter(llm, token_data):
    if llm == "gemini-2.0-flash":
        input_tokens_api = token_data.prompt_token_count
        output_tokens_api = token_data.candidates_token_count
        total_tokens_api = token_data.total_token_count
    elif llm == "claude-3-7-sonnet-20250219":
        input_tokens_api = token_data.input_tokens
        output_tokens_api = token_data.output_tokens
        total_tokens_api = token_data.input_tokens + token_data.output_tokens
    elif llm == "o3-mini":
        input_tokens_api = token_data["prompt_tokens"]
        output_tokens_api = token_data["completion_tokens"]
        total_tokens_api = token_data["total_tokens"]
    else:
        input_tokens_api = 0
        output_tokens_api = 0
        total_tokens_api = 0
    return input_tokens_api, output_tokens_api, total_tokens_api

def process_row(row, output_dir, api_key, api_key_verif, model_type, mode=0):
    """Process a single row from the dataset and return token usage info."""
    index = row.get('index', 0)
    question = row.get('question')
    solution = row.get('solution')
    updated_row = row.copy()

    # Write the question to a file in the output directory
    question_file_path = os.path.join(output_dir, "question.txt")
    try:
        with open(question_file_path, "w") as question_file:
            question_file.write(question)
        # print(f"Question written to {question_file_path}")
    except Exception as e:
        print(f"Failed to write question to file: {e}")
    
    if not question:
        print(f"Skipping row {index} due to missing question.")
        return None
    
    print(f"\nProcessing row {index}: Question: {question}")
    try:
        # Call the selected LLM API based on model_type
        if mode == 2:
            rerun_prompt = (
                f"Question: {row['question']}\n"
                f"Solution Ground Truth: {row['solution']}\n"
                f"Solution Candidate: {row.get('solution_candidate')}\n"
                f"Candidate Reasoning Plan: {row.get('candidate_reasoning_plan')}\n"
                f"Critique: {row.get('critique')}\n"
            )
            response, token_data = call_llm(model_type, api_key, prompt.REASONING_AFTER_REVIEW_PROMPT, rerun_prompt)
        else:
            response, token_data = call_llm(model_type, api_key, prompt.REASONING_STRUCTURE_PROMPT, question)

    except Exception as e:
        logger.error(f"LLM call failed for row {index}: {row}. Error: {e}")
        return None
 
    if not response:
        print(f"LLM response is empty for row {index}.")
        return None

    print("LLM Response:\n", response)
    print("Token Data:", token_data)

    # Determine token counts based on model type
    if model_type.lower() == "gemini":
        input_token_count = token_data.prompt_token_count
        output_token_count = token_data.candidates_token_count
        total_token_count = token_data.total_token_count
    else:  # Assume anthropic/sonnet-3.7 (or others with similar structure)
        input_token_count = token_data.input_tokens
        output_token_count = token_data.output_tokens
        total_token_count = token_data.input_tokens + token_data.output_tokens

    # Set up local token counts for aggregation
    local_tokens = {
        "input_tokens_api": input_token_count,
        "output_tokens_api": output_token_count,
        "total_tokens_api": total_token_count,
        "input_tokens_api_verif": 0,
        "output_tokens_api_verif": 0,
        "total_tokens_api_verif": 0,
        "input_tokens_api_reph": 0,
        "output_tokens_api_reph": 0,
        "total_tokens_api_reph": 0,
    }
    
    # Extract reasoning components
    reasoning_components = extract_reasoning_components(response)
    components_dict = reasoning_components.to_dict() if hasattr(reasoning_components, "to_dict") else reasoning_components
    
    # Update row with reasoning components
    updated_row = {**row, **components_dict}
    if mode == 1:
        pass
    else:
        try:
            # Save individual result to prevent data loss
            row_df = pd.DataFrame([updated_row])
            row_filename = os.path.join(output_dir, f"processed_row_{index}.csv")
            row_df.to_csv(row_filename, index=False)
        except Exception as e:
            print(f"Failed to save row {index}: {e}")

        return updated_row, local_tokens
    
    # Call verification if needed
    response_verif = None
    token_data_verif = None
    response_reph = None
    token_data_reph = None
    if components_dict.get("reasoning_plan") or components_dict.get("reasoning_trace"):
        verif_prompt = (
            f"Question: {row['question']}\n"
            f"Reasoning plan: {components_dict['reasoning_plan']}\n"
            f"Reasoning trace: {components_dict['reasoning_trace']}\n"
        )
        try:
            response_verif, token_data_verif = call_llm("o4-mini", api_key_verif, prompt.REASONING_VERIFIER_PROMPT, verif_prompt)
        except Exception as e:
            logger.error(f"Verifier call failed for row {index}: {e}")
        print("Verifier Response:\n", response_verif)
        if response_verif and token_data_verif:
            fixed_components = extract_verification_components(response_verif)
            fixed_dict = fixed_components.to_dict() if hasattr(fixed_components, "to_dict") else fixed_components
            updated_row = {**updated_row, **fixed_dict}
            print("Updated row with verification components for row", index)
            local_tokens["input_tokens_api_verif"] = token_data_verif.get("prompt_tokens", 0)
            local_tokens["output_tokens_api_verif"] = token_data_verif.get("completion_tokens", 0)
            local_tokens["total_tokens_api_verif"] = token_data_verif.get("total_tokens", 0)

            # If reasoning and solution are different, rerun with solution hints.
            comparison_result = None
            if solution and response_verif:
                comparison_result = cluster_results.compare_solutions(solution, fixed_dict['optimized_reasoning'], True)
                print(f"Comparison Result for row {index}: {comparison_result}")            

            if comparison_result is None:
                REPHRASE = False
                logger.info(f"Comparison result does not exist for row {index} {question}.")
            else:
                REPHRASE = not comparison_result[0]
                # REPHRASE = True #debugging. Force rephrase regardless of comparison result.
            print("REPHRASE:", REPHRASE)
            if REPHRASE:
                solution_hint_prompt = f"Question: {row['question']}\nSolution: {row['solution']}. Refer to the solution and solve yourself. Put your final answer in $\\boxed{{}}$"
                print("Rephrase Prompt:\n", solution_hint_prompt)
                response_reph, token_data_reph = call_llm(model_type, api_key_verif, prompt.REASONING_STRUCTURE_PROMPT, solution_hint_prompt)
                print("Rephrased Response:\n", response_reph)
                fixed_components = extract_solution_hint_components(response_reph)
                print("Extracted Rephrase Components:", fixed_components)
                fixed_dict = fixed_components.to_dict() if hasattr(fixed_components, "to_dict") else fixed_components
                updated_row = {**updated_row, **fixed_dict}

                print("Updated row with rephrasing components for row", index)
                print("Token Data Rephrase:", token_data_reph)
                token_in, token_out, token_total = token_counter("gemini-2.0-flash", token_data_reph)
                local_tokens["input_tokens_api_reph"] += token_in
                local_tokens["output_tokens_api_reph"] += token_out
                local_tokens["total_tokens_api_reph"] += token_total
        else:
            print(f"Verifier response is empty for row {index}.")

    # Save individual result to prevent data loss
    try:
        row_df = pd.DataFrame([updated_row])
        row_filename = os.path.join(output_dir, f"processed_row_{index}.csv")
        row_df.to_csv(row_filename, index=False)
    except Exception as e:
        print(f"Failed to save row {index}: {e}")
    
    return updated_row, local_tokens

def process_api_reasoning_parallel(dataset, api_keys, api_keys_verif, model_type, output_dir, mode=0):
    """
    Process the dataset in parallel, calling LLM APIs for reasoning and verification.
    Returns the updated dataset and aggregated token usage.
    """
    # Convert to pandas DataFrame if not already
    if not isinstance(dataset, pd.DataFrame):
        try:
            dataset = dataset.to_pandas()
        except Exception as e:
            print("Error converting dataset to DataFrame:", e)
            return dataset, None

    api_enabled = True
    if not api_enabled:
        return dataset, None

    # Define output file paths
    output_json_file = os.path.join(output_dir, "train_ds_math_updated.json")
    output_csv_file = os.path.join(output_dir, "train_ds_math_updated.csv")
    
    # Add index to dataset if not present
    if 'index' not in dataset.columns:
        dataset = dataset.reset_index(drop=True)
    
    # Convert to list of dictionaries
    rows = dataset.to_dict('records')

    if model_type == "gemini":
        num_workers = 50 # Tier1 rate. RPM 2,000, TPM 4M.  
    else: 
        num_workers = 10
        # num_workers = len(api_keys)  # Use the number of API keys as the max workers
    
    results = []
    temp_dirs = []  # Keep track of all temporary directories created
    aggregated_tokens = {
        "input_tokens_api": 0,
        "output_tokens_api": 0,
        "total_tokens_api": 0,
        "input_tokens_api_verif": 0,
        "output_tokens_api_verif": 0,
        "total_tokens_api_verif": 0,
        "input_tokens_api_reph": 0,
        "output_tokens_api_reph": 0,
        "total_tokens_api_reph": 0,
    }
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, row in enumerate(rows):
            # Create a unique temporary directory for each row
            dir = os.path.join(output_dir, "temp")
            os.makedirs(dir, exist_ok=True)  # Create the directory if it doesn't exist
            temp_dir = tempfile.mkdtemp(dir=dir, prefix=f"temp_row_{i}_")
            temp_dirs.append(temp_dir)  # Store the directory for later cleanup
            print(f"Temporary directory created for row {i}: {temp_dir}")

            # Select an API key in a round-robin manner
            key = api_keys[i % len(api_keys)]
            key_verif = api_keys_verif[i % len(api_keys_verif)]
            futures.append(executor.submit(process_row, row, temp_dir, key, key_verif, model_type, mode))
        
        # Collect results and aggregate token counts
        for future in futures:
            result = future.result()
            if result is not None:
                row_result, token_counts = result
                results.append(row_result)
                for k in aggregated_tokens:
                    aggregated_tokens[k] += token_counts.get(k, 0)
    
    # Create updated dataset and save
    updated_dataset = None
    if results:
        updated_dataset = pd.DataFrame(results)
        updated_dataset.to_json(output_json_file, orient='records', lines=True)
        updated_dataset.to_csv(output_csv_file, index=False)
        print("Processing complete. Results saved to disk.")
    else:
        print("No results processed successfully.")
    
    # Clean up temporary directories
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Temporary directory removed for {temp_dir}")
    
    return updated_dataset, aggregated_tokens

def process_api_reasoning(dataset, model_type, output_dir, mode=0):
    """
    Iterates over the combined dataset, generates reasoning via an LLM call,
    extracts components from the response, and if available, verifies the reasoning.
    Returns the updated dataset and aggregated token usage.
    """
    print("process_api_reasoning")
    
    api_enabled = True
    if not api_enabled:
        return dataset, None

    # Define output file paths for incremental saving
    output_json_file_api = os.path.join(output_dir, "train_ds_math_updated.json")
    output_csv_file_api = os.path.join(output_dir, "train_ds_math_updated.csv")

    # Token counters
    aggregated_tokens = {
        "input_tokens_api": 0,
        "output_tokens_api": 0,
        "total_tokens_api": 0,
        "input_tokens_api_verif": 0,
        "output_tokens_api_verif": 0,
        "total_tokens_api_verif": 0,
        "input_tokens_api_reph": 0,
        "output_tokens_api_reph": 0,
        "total_tokens_api_reph": 0,
    }

    for index, row in enumerate(tqdm.tqdm(dataset, desc="Processing dataset")):
        print(f"\nProcessing row {index}: {row}")
        question = row.get('question')
        solution = row.get('solution')
        if question:
            print(f"Question: {question}")
            if solution:
                print(f"Solution: {solution}")

            try:
                # For non-parallel mode, put api_key to None, to choose the default one
                api_key = None
                if mode == 2:
                    rerun_prompt = (
                        f"Question: {row['question']}\n"
                        f"Solution Ground Truth: {row['solution']}\n"
                        f"Solution Candidate: {row.get('solution_candidate')}\n"
                        f"Candidate Reasoning Plan: {row.get('candidate_reasoning_plan')}\n"
                        f"Critique: {row.get('critique')}\n"
                    )
                    response, token_data = call_llm(model_type, api_key, prompt.REASONING_AFTER_REVIEW_PROMPT, rerun_prompt)
                else:
                    response, token_data = call_llm(model_type, api_key, prompt.REASONING_STRUCTURE_PROMPT, question)
            except Exception as e:
                logger.error(f"LLM call failed for row {index}: {row}. Error: {e}")
                continue  # Skip to the next row if the call fails

            print("LLM Response:\n", response)
            print("Token Data:", token_data)

            # Determine token counts based on model type
            if model_type.lower() == "gemini":
                aggregated_tokens["input_tokens_api"] += token_data.prompt_token_count
                aggregated_tokens["output_tokens_api"] += token_data.candidates_token_count
                aggregated_tokens["total_tokens_api"] += token_data.total_token_count
            else:  # Assume anthropic/sonnet-3.7 (or others with similar structure)
                aggregated_tokens["input_tokens_api"] += token_data.input_tokens
                aggregated_tokens["output_tokens_api"] += token_data.output_tokens
                aggregated_tokens["total_tokens_api"] += token_data.input_tokens + token_data.output_tokens

            # Extract reasoning components
            reasoning_components = extract_reasoning_components(response)
            print("Extracted Reasoning Components:", reasoning_components)
            components_dict = reasoning_components.to_dict() if hasattr(reasoning_components, "to_dict") else reasoning_components

            # Update dataset for this row
            dataset = dataset.map(lambda example, idx: 
                                  {**example, **components_dict} if idx == index else example, 
                                  with_indices=True)
            print("Updated dataset with reasoning components for row", index)

            if mode == 1:
                pass
            else:
                # Incremental saving after any updates
                dataset.to_json(output_json_file_api, orient='records', lines=True)
                dataset.to_csv(output_csv_file_api, index=False)
                continue

            # If verification is needed
            if components_dict.get("reasoning_plan") and components_dict.get("reasoning_trace"):
                api_key_verif = os.getenv("OPENAI_API_KEY")
                verif_prompt = (
                    f"Question: {row['question']}\n"
                    f"Reasoning plan: {components_dict['reasoning_plan']}\n"
                    f"Reasoning trace: {components_dict['reasoning_trace']}\n"
                )

                response_verif, token_data_verif = call_llm("o4-mini", api_key_verif, prompt.REASONING_VERIFIER_PROMPT, verif_prompt)
                print("Verifier Response:\n", response_verif)
                fixed_components = extract_verification_components(response_verif)
                print("Extracted Verification Components:", fixed_components)
                fixed_dict = fixed_components.to_dict() if hasattr(fixed_components, "to_dict") else fixed_components

                # Update dataset for this row with verification components
                dataset = dataset.map(lambda example, idx: 
                                      {**example, **fixed_dict} if idx == index else example, 
                                      with_indices=True)
                print("Updated dataset with verification components for row", index)

                aggregated_tokens["input_tokens_api_verif"] += token_data_verif["prompt_tokens"]
                aggregated_tokens["output_tokens_api_verif"] += token_data_verif["completion_tokens"]
                aggregated_tokens["total_tokens_api_verif"] += token_data_verif["total_tokens"]

                # If reasoning and solution are different, rerun with solution hints.
                comparison_result = None
                if solution and response_verif:
                    comparison_result = cluster_results.compare_solutions(solution, fixed_dict['optimized_reasoning'], True)
                    print(f"Comparison Result for row {index}: {comparison_result}")

                if comparison_result is None:
                    REPHRASE = False
                    logger.info(f"Comparison result does not exist for row {index} {question}.")
                else:
                    REPHRASE = not comparison_result[0]
                    # REPHRASE = True #debugging. Force rephrase regardless of comparison result.
                print("REPHRASE:", REPHRASE)
                if REPHRASE:
                    solution_hint_prompt = f"Question: {row['question']}\nSolution: {row['solution']}. Refer to the solution and solve yourself. Put your final answer in $\\boxed{{}}$"
                    print("Rephrase Prompt:\n", solution_hint_prompt)
                    response_reph, token_data_reph = call_llm(model_type, None, prompt.REASONING_STRUCTURE_PROMPT, solution_hint_prompt)
                    print("Rephrased Response:\n", response_reph)
                    fixed_components = extract_solution_hint_components(response_reph)
                    print("Extracted Rephrase Components:", fixed_components)
                    fixed_dict = fixed_components.to_dict() if hasattr(fixed_components, "to_dict") else fixed_components

                    # Update dataset for this row with rephrase components
                    dataset = dataset.map(lambda example, idx: 
                                        {**example, **fixed_dict} if idx == index else example, 
                                        with_indices=True)
                    print("Updated dataset with rephrase components for row", index)
                    token_in, token_out, token_total = token_counter("gemini-2.0-flash", token_data_reph)
                    aggregated_tokens["input_tokens_api_reph"] += token_in
                    aggregated_tokens["output_tokens_api_reph"] += token_out
                    aggregated_tokens["total_tokens_api_reph"] += token_total

            # Incremental saving after any updates
            dataset.to_json(output_json_file_api, orient='records', lines=True)
            dataset.to_csv(output_csv_file_api, index=False)

    print("process_api_reasoning  {idx} complete. Results saved to disk.", dataset)
    return dataset, aggregated_tokens

def print_token_summary(tokens, model_type="anthropic"):
    """
    Print the token summary for API calls.
    For inference cost, it uses the appropriate label based on the model_type.
    """
    print("\nToken Summary (API):")
    print("Total input tokens:", tokens["input_tokens_api"])
    print("Total output tokens:", tokens["output_tokens_api"]) 
    print("Total tokens:", tokens["total_tokens_api"])
    
    # Choose the cost label based on model_type
    if model_type.lower() == "gemini":
        cost_label = "gemini-2.0-flash"
    elif model_type.lower() == "anthropic":
        cost_label = "claude-3.7-sonnet"
    elif model_type.lower() == "qwq":
        cost_label = "qwq-32b-preview"
    else:
        cost_label = "claude-3.7-sonnet"
    
    print("Inference cost:", estimate_cost.api_cost_estimation_service(
        tokens["input_tokens_api"], tokens["output_tokens_api"], cost_label))
    
    print("\nTotal input tokens verif:", tokens["input_tokens_api_verif"])
    print("Total output tokens verif:", tokens["output_tokens_api_verif"])
    print("Total tokens verif:", tokens["total_tokens_api_verif"])
    print("Verification cost:", estimate_cost.api_cost_estimation_service(
        tokens["input_tokens_api_verif"], tokens["output_tokens_api_verif"], "o3-mini"))
    
    print("\nTotal input tokens reph:", tokens["input_tokens_api_reph"])
    print("Total output tokens reph:", tokens["output_tokens_api_reph"])
    print("Total tokens reph:", tokens["total_tokens_api_reph"])
    print("rephrase cost:", estimate_cost.api_cost_estimation_service(
        tokens["input_tokens_api_reph"], tokens["output_tokens_api_reph"], cost_label))