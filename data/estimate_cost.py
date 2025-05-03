
pricing = [
        {'provider': 'openai', 'model': 'gpt-4o', 'cost_per_million_input': 2.50, 'cost_per_million_output': 10.00},
        {'provider': 'openai', 'model': 'gpt-4o-mini', 'cost_per_million_input': 0.15, 'cost_per_million_output': 0.60},
        {'provider': 'openai', 'model': 'o1', 'cost_per_million_input': 15.00, 'cost_per_million_output': 60.00},
        {'provider': 'openai', 'model': 'o4-mini', 'cost_per_million_input': 1.10, 'cost_per_million_output': 4.40},
        {'provider': 'openai', 'model': 'o3-mini', 'cost_per_million_input': 1.10, 'cost_per_million_output': 4.40},
        {'provider': 'openai', 'model': 'o1-mini', 'cost_per_million_input': 3.00, 'cost_per_million_output': 15.00},
        {'provider': 'anthropic', 'model': 'claude-3.7-sonnet', 'cost_per_million_input': 3.00, 'cost_per_million_output': 15.00},
        {'provider': 'anthropic', 'model': 'claude-3.5-haiku', 'cost_per_million_input': 0.80, 'cost_per_million_output': 4.00},
        {'provider': 'meta', 'model': 'llama-3.3-70b-versatile-128k', 'cost_per_million_input': 0.59, 'cost_per_million_output': 0.79},
        {'provider': 'google', 'model': 'gemini-2.5-pro-preview', 'cost_per_million_input': 1.25, 'cost_per_million_output': 10.00},
        {'provider': 'google', 'model': 'gemini-2.0-flash', 'cost_per_million_input': 0.10, 'cost_per_million_output': 0.40},
        {'provider': 'deepseek', 'model': 'chat', 'cost_per_million_input': 0.27, 'cost_per_million_output': 1.10},
        {'provider': 'deepseek', 'model': 'chat', 'cost_per_million_input': 0.55, 'cost_per_million_output': 2.19},
        {'provider': 'groq', 'model': 'qwen-2.5-coder-32b', 'cost_per_million_input': 0.79, 'cost_per_million_output': 0.79},
        {'provider': 'groq', 'model': 'qwq-32b-preview', 'cost_per_million_input': 0.29, 'cost_per_million_output': 0.39}
   ]

def get_pricing_by_model(model_name):
    for entry in pricing:
        if entry['model'] == model_name:
            return entry
    return None

def api_cost_estimation(total_input_tokens, total_output_tokens):
   # API Pricing details for various models
   # Calculate and print the cost for input and output tokens for each model
   print(f'Cost estimates based on API pricing:\n')
   for entry in pricing:
        cost_input = (total_input_tokens / 1_000_000) * entry['cost_per_million_input']
        cost_output = (total_output_tokens / 1_000_000) * entry['cost_per_million_output']
        total_cost = cost_input + cost_output
        print(f'Provider: {entry["provider"]}, Model: {entry["model"]}')
        print(f'  Input cost: ${cost_input:.4f}')
        print(f'  Output cost: ${cost_output:.4f}')
        print(f'  Total cost: ${total_cost:.4f}\n')
   print(f'input tokens: {total_input_tokens}')
   print(f'output tokens: {total_output_tokens}\n')

def api_cost_estimation_service(total_input_tokens, total_output_tokens, service):
    entry = get_pricing_by_model(service)
    print(f'Provider: {entry["provider"]}, Model: {entry["model"]}')
    cost_input = (total_input_tokens / 1_000_000) * entry['cost_per_million_input']
    cost_output = (total_output_tokens / 1_000_000) * entry['cost_per_million_output']
    total_cost = cost_input + cost_output
    print(f'input tokens: {total_input_tokens}')
    print(f'output tokens: {total_output_tokens}\n')
    print(f'  Input cost: ${cost_input:.4f}')
    print(f'  Output cost: ${cost_output:.4f}')
    print(f'  Total cost: ${total_cost:.4f}\n')
    return total_cost

if __name__ == "__main__":
    # Example token counts
    total_input_tokens = 10000000  # Replace with actual input token count
    total_output_tokens = 40000000  # Replace with actual output token count

    # Call the function to estimate costs
    api_cost_estimation(total_input_tokens, total_output_tokens)