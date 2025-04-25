import re

def extract_component(text, start_markers, end_markers=None):
    """
    Extracts the substring from 'text' that is found between any start marker and any end marker.
    If no end_markers are provided, returns text after the first matching start marker.
    """
    for start in start_markers:
        if start in text:
            start_idx = text.find(start) + len(start)
            if end_markers:
                for end in end_markers:
                    end_idx = text.find(end, start_idx)
                    if end_idx != -1:
                        return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:].strip()
    return None

def extract_solution_hint_components(response):
    """
    Extracts the reasoning plan and trace from the LLM response.
    Looks for plan markers and then extracts the text after the plan.
    """
    components = {}
    components["solution_hint_plan"] = extract_component(
        response,
        start_markers=["<|begin_of_plan|>"],
        end_markers=["<|end_of_plan|>","|end_of_plan|"] #Sometimes llm miss '<'
    )
    # For reasoning trace, take the text after the first encountered end marker.
    for marker in ["<|end_of_plan|>","|end_of_plan|"]:
        if marker in response:
            components["solution_hint_reasoning"] = response.split(marker, 1)[1].strip()
            break
    else:
        components["solution_hint_reasoning"] = None
    return components

def extract_reasoning_components(response):
    """
    Extracts the reasoning plan and trace from the LLM response.
    Looks for plan markers and then extracts the text after the plan.
    """
    components = {}
    components["reasoning_plan"] = extract_component(
        response,
        start_markers=["<|begin_of_plan|>"],
        end_markers=["<|end_of_plan|>","|end_of_plan|"] #Sometimes llm miss '<'
    )
    # For reasoning trace, take the text after the first encountered end marker.
    for marker in ["<|end_of_plan|>","|end_of_plan|"]:
        if marker in response:
            components["reasoning_trace"] = response.split(marker, 1)[1].strip()
            break
    else:
        components["reasoning_trace"] = None
    return components

def extract_verification_components(response):
    """
    Extracts fixed and optimized reasoning components from the verifier response,
    including fixed and optimized reasoning (if provided) and the verification score.
    """
    components = {}
    # Extract fixed plan and trace.
    components["fixed_plan"] = extract_component(
        response,
        start_markers=["<fixed_plan>"],
        end_markers=["</fixed_plan>"]
    )
    # Extract fixed and optimized reasoning.
    components["fixed_reasoning"] = extract_component(
        response,
        start_markers=["<fixed_reasoning>"],
        end_markers=["</fixed_reasoning>"]
    )

    # Extract optimized plan and trace.
    components["optimized_plan"] = extract_component(
        response,
        start_markers=["<optimized_plan>"],
        end_markers=["</optimized_plan>"]
    )
    components["optimized_reasoning"] = extract_component(
        response,
        start_markers=["<optimized_reasoning>"],
        end_markers=["</optimized_reasoning>"]
    )

    # Extract verification score in fraction format, e.g. {{4/4}}, {{6/7}}, etc.
    match = re.search(r"\{\{(\d+/\d+)\}\}", response)
    # [[]]
    # match = re.search(r"\[\[(\d+/\d+)\]\]", response)
    if match:
        # 100 pct representation
        # components["reasoning_step_score"] = eval(match.group(1)) * 100
        components["reasoning_step_score"] = match.group(1)
    else:
        components["reasoning_step_score"] = None

    return components

def extract_rephrase_components(response):
    # "rephrased_question": "",
    # "rephrased_plan": "",
    # "rephrased_reasoning": "",
    components = {}
    components["rephrased_question"] = extract_component(
        response,
        start_markers=["<rephrased_question>"],
        end_markers=["</rephrased_question>"]
    )
    components["rephrased_plan"] = extract_component(
        response,
        start_markers=["<rephrased_plan>"],
        end_markers=["</rephrased_plan>"]
    )
    components["rephrased_reasoning"] = extract_component(
        response,
        start_markers=["<rephrased_reasoning>"],
        end_markers=["</rephrased_reasoning>"]
    )

    return components