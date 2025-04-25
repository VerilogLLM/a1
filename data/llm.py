
import tiktoken
import time

from openai import OpenAI
TEMPERATURE = 0.8

def count_tokens(text, model):
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text into tokens
    tokens = encoding.encode(text)
    # Return both the list of tokens and the total count
    return tokens, len(tokens)

def llm_vllm(model, system_prompt, user_prompt):
    start_time = time.time()  # Start the timer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        # Construct a combined prompt from the message list.
        prompt_text = ""
        for message in messages:
            prompt_text += f"{message['role']}: {message['content']}\n"
        
        # Here we assume a vllm interface similar to other LLM APIs.
        # You might need to adjust this depending on your actual vllm setup.
        from vllm import LLMEngine  # Ensure the vllm library is installed and imported.
        
        # Initialize the engine with the specified model.
        engine = LLMEngine(model=model)
        
        # Generate a completion. Adjust parameters as needed.
        response = engine.generate(
            prompt=prompt_text,
            max_tokens=8192,  # Adjust as necessary.
            temperature=TEMPERATURE
        )
        
        # Assuming the response object has a 'text' attribute with the output.
        end_time = time.time()  # End the timer
        print(f"\nllm_vllm Execution time: {end_time - start_time:.2f} seconds")
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response for prompt: {user_prompt}\nException: {e}")
        end_time = time.time()  # End the timer
        print(f"\nllm_vllm Execution time: {end_time - start_time:.2f} seconds")
        return ""

def llm_openai(model, system_prompt, user_prompt):
    start_time = time.time()  # Start the timer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            store=True,
            max_tokens=8192,  # Adjust as needed
            temperature=0.3
        )
        # Get token counts
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        token_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        end_time = time.time()  # End the timer
        print(f"\nllm_openai Execution time: {end_time - start_time:.2f} seconds")
        return response.choices[0].message.content.strip(), token_data
    except Exception as e:
        print(f"Error generating response for prompt: {user_prompt}\nException: {e}")
        end_time = time.time()  # End the timer
        print(f"\nllm_openai Execution time: {end_time - start_time:.2f} seconds")
        return ""

def llm_openai(model, system_prompt, user_prompt, api_key):
    start_time = time.time()  # Start the timer
    # System message get ignored.
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt}
    # ]
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    messages = [
        {"role": "user", "content": combined_prompt}
    ]
    
    try:
        client = OpenAI(api_key=api_key)
        # store=True,
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort="high",
            max_completion_tokens=100000,  # Adjust as needed
        )
        # Get token counts
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        token_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        end_time = time.time()  # End the timer
        print(f"\nllm_openai {model} Execution time: {end_time - start_time:.2f} seconds")
        return response.choices[0].message.content.strip(), token_data
    except Exception as e:
        print(f"Error generating response for prompt: {user_prompt}\nException: {e}")
        end_time = time.time()  # End the timer
        print(f"\nllm_openai {model} Execution time: {end_time - start_time:.2f} seconds")
        return ""

from ollama import chat
from ollama import ChatResponse
def llm_ollama(model, system_prompt, user_prompt):
    start_time = time.time()  # Start the timer
    try:
        response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'system',
                'content': system_prompt,
                'role': 'user',
                'content': user_prompt,
            },
            ])
        end_time = time.time()  # End the timer
        print(f"\nllm_ollama Execution time: {end_time - start_time:.2f} seconds")
        return response.message.content.strip()
    except Exception as e:
        print(f"Error generating response for prompt: {user_prompt}\nException: {e}")
        end_time = time.time()  # End the timer
        print(f"\nllm_ollama Execution time: {end_time - start_time:.2f} seconds")
        return ""

#
# Anthropic API
#

from anthropic import Anthropic
# import anthropic
# client = anthropic.Anthropic(
#     # defaults to os.environ.get("ANTHROPIC_API_KEY")
#     api_key="my_api_key",
# )
# message = client.messages.create(
#     model="claude-3-7-sonnet-20250219",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, Claude"}
#     ]
# )
# print(message.content)

def llm_anthropic(model, system_prompt, user_prompt, api_key):
    start_time = time.time()  # Start the timer
    try:
        # client = Anthropic()
        client = Anthropic(
            api_key=api_key,
        )
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=20000,
        )
        # print(response)
        # print(response.content[0].text.strip())
        end_time = time.time()  # End the timer
        print(f"\nllm_anthropic Execution time: {end_time - start_time:.2f} seconds")
        return response.content[0].text.strip(), response.usage
    except Exception as e:
        print(f"Error generating response for prompt: {user_prompt}\nException: {e}")
        end_time = time.time()  # End the timer
        print(f"\nllm_anthropic Execution time: {end_time - start_time:.2f} seconds")
        return ""    

# def llm_anthropic_ext_think(model, system_prompt, user_prompt):
#     start_time = time.time()  # Start the timer
#     try:
#         client = Anthropic()
#         response = client.messages.create(
#             model=model,
#             system=system_prompt,
#             messages=[{"role": "user", "content": user_prompt}],
#             max_tokens=20000,
#             thinking={
#                 "type": "enabled",
#                 "budget_tokens": 16000,
#             }
#         )
#         # print(response) #returns response.content[ThinkingBlock, TextBlock]
#         return response.content, response.usage
#     except Exception as e:
#         print(f"Error generating response for prompt: {user_prompt}\nException: {e}")
#         return ""  

from google import genai
from google.genai import types
import os
def llm_gemini(model, system_prompt, user_prompt, api_key=None):
    start_time = time.time()  # Start the timer
    if api_key is None:
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    else:
        client = genai.Client(
            api_key=api_key,
        )
    # model = "gemini-2.5-pro-exp-03-25"
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="text/plain",
            temperature=TEMPERATURE,
            max_output_tokens=100000,
        )
    )
    print("response:", response)
    end_time = time.time()  # End the timer
    print(f"\nllm_gemini Execution time: {end_time - start_time:.2f} seconds")
    return response.text.strip(), response.usage_metadata

def llm_gemini_stream(model, system_prompt, user_prompt):
    start_time = time.time()  # Start the timer
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    # model = "gemini-2.5-pro-exp-03-25"
    model = model
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    text_stream = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        text_stream = text_stream + " " + chunk.text
    
    end_time = time.time()  # End the timer
    print(f"\nllm_gemini Execution time: {end_time - start_time:.2f} seconds")
    return text_stream.strip()