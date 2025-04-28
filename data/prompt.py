import json

from datasets import Dataset

### REFERENCE ###
### REFERENCE ###

# Plan and action are aligned.
REASONING_STRUCTURE_PROMPT = """
Your role as an assistant is to approach question by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach. After <|end_of_plan|> Loop through each step in the plan, and perform. After final step, conclude your answer. When answer can't be concluded, determine if the user question is valid.
Keep supportive statement as brief as possible. Put your final answer in boxed format \\boxed{{answer}}$ where [answer] is just the final number or expression that solve the problem.
Do not repeat the same sentence in reasoning.
"""

# Evaluation of reasoning trace by o3. Print reasoning fix. structured output of fixed plan and reasoning. Keep it simple.
REASONING_VERIFIER_PROMPT = """
You are provided with a question, a plan to approach, and detailed reasoning trace. 
First, solve a given question yourself and keep it for a reference.
Your task are the followings.
1) Compare your solution with the given detailed reasoning trace.
2) Check if given plan and given reasoning trace are aligned. Loop for each step, if you think a step is aligned with the plan, and reasoning makes sense then score it as 1. If you think either a step is not aligned with the plan, or reasoning is not right then score it as 0.
3) If a step is wrong or scored as 0, fix the step.
4) If you fixed a step, provide a structured output of fixed plan and reasoning in <fixed_plan> and <fixed_reasoning> tags.
5) Finally, optimize plan and reasoning for the most optimized answer generation process. Remove redundant steps and reasoning. Remove unnecessary words. Write in <optimized_plan> and <optimized_reasoning> tags. In <optimized_reasoning> tag, put your final answer in boxed format \\boxed{{answer}}$ where [answer] is just the final number or expression that solve the problem.

Put the verification score (counts aligned)/(plan counts) in fraction format in the format {{}}.
Do not repeat the same sentence in reasoning.
"""


### TRIAL ###
### TRIAL ###

REASONING_AFTER_REVIEW_PROMPT = """
You are an advanced reasoning assistant tasked with solving mathematical and logical problems. For each question, you are provided with the following:

1. question: The problem statement that needs to be solved.
2. solution(ground truth): The correct solution to the problem, which serves as a reference.
3. solution candidate: A proposed solution to the problem, which may or may not be correct.
4. critique: An analysis of the solution candidate, highlighting whether it aligns with the ground truth solution and identifying any errors or discrepancies.

### Your Task:
- Solve the question.
- Carefully analyze the **question**, **ground truth solution**, **solution candidate**, and **critique**.
- Use the critique as a guide to solve the question. 
- In most cases, ground truth solution is correct. If your solution does not match the ground truth solution, check if critique makes sense.
- Ensure your reasoning is clear, concise, and logically sound.
- approach question by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach. After <|end_of_plan|> Loop through each step in the plan, and perform. After final step, conclude your answer. When answer can't be concluded, determine if the user question is valid.
Keep supportive statement as brief as possible. Put your final answer in boxed format \\boxed{{answer}}$ where [answer] is just the final number or expression that solve the problem. Do not repeat the same sentence in reasoning.
"""





### TRASH ###
### TRASH ###

# Improves nothing.
# REPHRASE_PROMPT = """
# You are provided with a question, a plan to approach, and detailed reasoning trace. 
# Given plan, rephrase the question to start reasoning from the middle of the plan.
# Collect the corresponding result from reasoning trace and feed it to the new question as conditions or hints.
# Write the new question in <rephrased_question> tag.

# With the rephrased question, 
# 1) Plan to approach the question. Write the new plan in <rephrased_plan> tag.
# 2) Perform the reasoning trace. Write the new reasoning in <rephrased_reasoning> tag. Put your final answer in boxed format {{}} in <rephrased_reasoning> tag.
# """


# # Evaluation of reasoning trace by o3. Print reasoning fix.
# REASONING_VERIFIER_PROMPT = """
# You are provided with a question, a plan to approach, and detailed reasoning trace. 
# First, solve a given question yourself and keep it for a reference.
# Your task are the followings.
# 1) Compare your solution with the given detailed reasoning trace.
# 2) Check if given plan and given reasoning trace are aligned. Loop for each step, if you think a step is aligned with the plan, and reasoning makes sense then score it as 1. If you think either a step is not aligned with the plan, or reasoning is not right then score it as 0.
# 3) If a step is wrong or scored as 0, fix the step.

# Put the verification score (counts aligned)/(plan counts) in fraction format in \boxed{}.
# """

# # Evaluation of reasoning trace by o3
# REASONING_VERIFIER_PROMPT = """
# You are provided with a question, a plan to approach, and detailed reasoning trace. 
# First, solve a given question yourself and keep it for a reference.
# Your task are the followings.
# 1) Compare your solution with the given detailed reasoning trace.
# 2) Check if given plan and given reasoning trace are aligned. Loop for each step, if you think a step is aligned with the plan, and reasoning makes sense then score it as 1. If you think either a step is not aligned with the plan, or reasoning is not right then score it as 0. For example, plan consists of 5 steps. Full score is 5. If three steps are aligned and correct, then final score is 3 out of 5. 

# Put the verification score 3/5 (fraction format) in \boxed{}.
# """


# Conclude at final step. At most one verification. More efficient and accurate.
# REASONING_STRUCTURE_PROMPT = """
# Your role as an assistant is to approach question by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach. Perform each step. After final step, conclude your answer. When answer can't be concluded, determine if the user question is valid.
# Keep supportive statement as brief as possible.
# """

# # Conclude at final step. At most one verification. Sometimes fall into a wrong branch and long wasteful reasoning.
# REASONING_STRUCTURE_PROMPT = """
# Your role as an assistant is to approach question by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach, including planning, verifying each step. After final step, conclude your answer. Be cautious of an invalid question.
# Keep supportive statement as brief as possible.
# """

# when answer can't be concluded, retry token length is too long.
# REASONING_STRUCTURE_PROMPT = """
# Your role as an assistant is to approach question by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach, including planning, verifying each step. 
# Following this, you engage in an in-depth internal reasoning process—captured within <|begin_of_thought|> and <|end_of_thought|>. 
# After final step, conclude your answer, do not perform extra verification. Be cautious of an invalid question.
# Keep supportive statement as brief as possible.
# Provide a solution in <|begin_of_solution|> and <|end_of_solution|>.
# """

# Extra verification doubles the tokens.
# REASONING_STRUCTURE_PROMPT = """
# Your role as an assistant is to approach question by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach, including planning, verifying each step. 
# Following this, you engage in an in-depth internal reasoning process—captured within <|begin_of_thought|> and <|end_of_thought|>. 
# After final step, conclude your answer enclosed within <|begin_of_solution|> and <|end_of_solution|>. Be cautious of an invalid question.
# Keep supportive statement as brief as possible.
# """


# cause long wasteful reasoning.
# REASONING_STRUCTURE_PROMPT = """
# “Your role as an assistant is to approach every query by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach, including planning, verifying each step, branching into alternative methods when errors are detected, and incorporating back proof. Following this, you engage in an in-depth internal reasoning process—captured within <|begin_of_thought|> and <|end_of_thought|>—that documents the iterative verification, exploration of alternate approaches, and refinement of your reasoning. Finally, you provide a clear, concise, and logically sound final solution enclosed within <|begin_of_solution|> and <|end_of_solution|>.”
# """

# cause long wasteful reasoning.
# REASONING_STRUCTURE_PROMPT = """
# Your role as an assistant is to approach every query by first constructing a comprehensive plan—enclosed within <|begin_of_plan|> and <|end_of_plan|>—that details your intended approach, including planning, verifying each step, branching into alternative methods when errors are detected, and incorporating back proof. Following this, you engage in an in-depth internal reasoning process—captured within <|begin_of_thought|> and <|end_of_thought|>—that documents the iterative verification, exploration of alternate approaches. If answer can't be concluded, try over only once. For math operation, write a code to generate a result. Finally, you provide a clear, concise, and logically sound final solution enclosed within <|begin_of_solution|> and <|end_of_solution|>.
# """

# bad. Other than telling each step is essential, it doesn't help.
# REASONING_VERIFIER_PROMPT = """
# You are provided with a detailed LLM reasoning trace and its final answer. Your task is to:
# 	1.	Split the entire text into individual sentences.
# 	2.	For each sentence, assess whether it contributes directly to deriving or supporting the correct final answer.
# 	3.	For each sentence, provide a brief explanation of its role—if it is essential, supportive, or irrelevant to the correctness of the final answer.
# 	4.	Conclude with a summary evaluation of the overall reasoning trace, indicating whether all sentences are necessary and correctly aligned with the final answer.
#     Ensure that your analysis is clear, objective, and focuses solely on the contribution of each sentence to achieving the correct answer.
# """



DEEPSEEK_R1_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered
thinking process.
"""

SKY_T1_SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"  # noqa


def format_code_prompt(x):
    formatted_prompt = ""

    data = json.loads(x["test_cases"])
    if not data.get("fn_name"):
        formatted_prompt += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # noqa
    else:
        formatted_prompt += (
            "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # noqa
        )

    formatted_prompt += x["problem"]
    if x["starter_code"] is not None:
        data = x["starter_code"]
        data = "\n" + data
        formatted_prompt += data
    return formatted_prompt


def map_to_share_gpt(x):
    if x["domain"] == "code" and "formatted_prompt" not in x:
        user = format_code_prompt(x)
    elif x["domain"] == "math":
        user = f"Return your final response within \\boxed{{}}. {x['question']}"
    else:
        user = x["question"]

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user},
            {
                "from": "assistant",
                "value": f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>",
            },
        ],
    }


def map_numina_conversations(x):
    """Map the Numina dataset to the required format."""
    user_message = f"Return your final response within \\boxed{{}}. {x['problem']}"
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )
    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


def apply_numina_map(dataset: Dataset) -> Dataset:
    numina_conversations = dataset.map(map_numina_conversations)
    return numina_conversations


def map_apps_conversations(x):
    """Map the APPS dataset to the required format."""
    test_case = json.loads(x["input_output"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n" #noqa
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n" #noqa
    data = prompt
    user_message += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


def apply_apps_map(dataset: Dataset) -> Dataset:
    apps_conversations = dataset.map(map_apps_conversations)
    return apps_conversations


def map_taco_conversations(x):
    """Map the TACO dataset to the required format."""
    test_case = json.loads(x["input_output_x"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n" #noqa
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n" #noqa
    data = prompt
    user_message += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


def apply_taco_map(dataset: Dataset) -> Dataset:
    taco_conversations = dataset.map(map_taco_conversations)
    return taco_conversations


def map_still2_conversations(x):
    """Map the still2 dataset to the required format."""
    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": x["question"]},
            {"from": "assistant", "value": x["combined_text"]},
        ],
    }


def apply_still2_map(dataset: Dataset) -> Dataset:
    still2_conversations = dataset.filter(lambda x: x["domain"] in ["puzzle", "physics", "biology", "chemistry"]).map(map_still2_conversations)
    return still2_conversations
