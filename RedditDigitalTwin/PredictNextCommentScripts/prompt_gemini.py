import sys
sys.path.append('../')
import GoogleGemini as gemini
import json

def prompt_llm(prompt, debug=True):
    """Prompts an LLM to identify if there is a complaint in text (Yes or No), 
    returns list with all posts that are a complaint"""

    print("Filtering posts...")
    if(debug):
        print("DEBUG ON")
    
    gemini.InitGoogleGemini()

    try:
        # print(f'Prompt: {prompt} \n\n') # Uncomment to see prompt
        response = gemini.AskGoogleGemini(prompt, force=False)
        # print(f'Response: {response} \n\n') # Uncomment to see LLM output
        # response = response.replace('\n', '')
        # Format as JSON
        startIndex = response.find('{')
        endIndex = response.find('}')+1
        response = response[startIndex:endIndex]
        response = json.loads(response)

    except Exception as e:
        print(e)
        # print(f'Blocked post: {posts}')

    return response 


