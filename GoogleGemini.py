import os
from google import genai 
from google.genai import types
import hashlib
import json

def InitGoogleGemini(folder='', free_tier=False):
    "Retrieve my API key and initialize Gemini with it"

    global client
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    if free_tier:
        with open(folder + 'MyPersonalKeyAPI/free_secret', 'r') as f: # Path to the free tier API key
            api_key = f.readline()
    else:
        with open(folder + 'MyPersonalKeyAPI/secret', 'r') as f: # Path to the API key
            api_key = f.readline()

    os.environ["GOOGLE_API_KEY"] = api_key
    client = genai.Client(api_key=api_key)
        

def AskGoogleGemini(prompt: str, model='gemini-3-flash-preview', max_output_tokens=4096, force=False, temperature=0.3, top_k=40) -> str:
    "Ask a prompt to given Google Cloud model and return the response text and safety ratings."

    # Check if the prompt has been executed before
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script

    response = ''
    hashedPrompt = str(hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8])
    filepath = folder + 'GooglegeminiCache/' + model + '/' + hashedPrompt
    if(os.path.isfile(filepath)):
        with open(filepath, 'r') as f:
            response = f.read()

    # If force is True, always get a new response from Gemini
    if(force):
        response = ''
       
    # Get the response and its safety ratings from Gemini if it was not cached
    if(response == ''):
        completion = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature, # Randomness: Low temp = low randomness, high temp = high creativity
                top_k=top_k,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        # Add other safety categories and thresholds here....
                        ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        # Add other safety categories and thresholds here....
                        ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        # Add other safety categories and thresholds here....
                        ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        # Add other safety categories and thresholds here....
                        ),
                    ]
            )
        )
        response = completion.text
        if(response is None):
            response = 'unknown'
        
        # Output the response and its safety ratings to cache if it has not been executed before
        with open(filepath, 'w') as f:
            f.write(response)
    
    return response


