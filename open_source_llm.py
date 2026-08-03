import pandas as pd
import os
import hashlib
import warnings
from huggingface_hub import InferenceClient
import time

warnings.filterwarnings("ignore")

def get_API_key(folder=''):
    folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    with open(folder + 'MyPersonalKeyAPI/huggingface', 'r') as f:
        api_key = f.readline()
        api_key = api_key.strip()
    return api_key


def make_inference_call(prompt, model_name, temperature=0.3, force=False):
    """Prompts llm to classify one post. 
    Input: a post and prompt,
    returns the output text.
    """

    # Get API key
    api_key = get_API_key()

    # Set location for cache
    folder = os.path.dirname(os.path.abspath(__file__)) + '/OpenSourceLLMsCache/'
    cache_dir = folder + model_name.replace('/', '__') + '_cache'
    os.makedirs(cache_dir, exist_ok=True)


    # Hash current prompt
    hashed_prompt = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
    filepath = cache_dir + '/' + hashed_prompt

    result_text = ''

    # Check if output already exists
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            result_text = f.read()

    if force:
        result_text = ''

    if result_text == '':
        client = InferenceClient(api_key=api_key)

        messages = [{"role": "user", "content": prompt}]

        while True:
            try:
                final_output = ""
                output = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=2048
                )

                # Collect the output in chunks
                for chunk in output: 
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        final_output += content

                with open(filepath, 'w') as f:
                    f.write(final_output)

                return final_output

            except Exception as e:
                print(f"API call failed: {e}")
                print("Waiting 30 seconds before retrying...")
                time.sleep(30)

        
    return result_text
