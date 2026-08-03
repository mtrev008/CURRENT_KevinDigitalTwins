import pandas as pd
import json
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()

def GetNumTopicsPerPost(user, posts, force=False, debug=False):
    """Takes in a key, val (user, all user posts)"""

    curr_user_posts_num_topics = {}

    # Iterate through each users posts
    for i, post in enumerate(posts[:5]):
        if debug:
            print(f"Post {i}...")

        # Get the X # of topics per post per user
        num_topics_prompt = ""
        num_topics_prompt += "I will provide you a Reddit comment. You will tell me how many topics are explicitly mentioned within the comment. "
        num_topics_prompt += "Format your answer as a single number with no other output. If the comment does not directly mention any specific topics, output 0. "
        num_topics_prompt += "The comment is:\n"
        num_topics_prompt += f'"""{post}"""'

        if debug: 
            print("Prompt:", num_topics_prompt)

        num_topics = gemini.AskGoogleGemini(num_topics_prompt, force=False)
        try:
            num_topics = int(num_topics)
        except:
            print(f"Could not parse number of topics. Output was: {num_topics}")
            num_topics = 0
            
        if debug:
            print("Number of topics:", num_topics)

        curr_user_posts_num_topics[post] = num_topics # Add post with it's number of topics to dict

    return curr_user_posts_num_topics

def TEMP_GetNumTopicsFromPost(post: str, force=False, debug=False) -> int:
    "Returns the number of topics mentioned in the given post"

    # Get the X # of topics per post per user
    num_topics_prompt = ""
    num_topics_prompt += "I will provide you with a Reddit comment. You will tell me how many topics are explicitly mentioned within the comment. "
    num_topics_prompt += "Answer only as a single number. If the comment does not directly mention any specific topics, output 0. "
    num_topics_prompt += "The comment is:\n"
    num_topics_prompt += f'"""{post}"""'

    if debug: 
        print("Prompt:", num_topics_prompt)

    num_topics = gemini.AskGoogleGemini(num_topics_prompt, force=False)
    num_topics = int(num_topics)
    if debug:
        print("Number of topics:", num_topics)

    return num_topics