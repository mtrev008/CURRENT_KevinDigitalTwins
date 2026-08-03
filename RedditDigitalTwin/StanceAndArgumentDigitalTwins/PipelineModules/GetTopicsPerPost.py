import pandas as pd
import json
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()


def GetTopicsPerPost(curr_user_posts_num_topics, force=False, debug=False):
    """Gets X number of topics per post, depending on the number of topics found in previous step"""
    curr_user_posts_topics = {}

    # Label each post with a topic out of list of topics per user
    for i, (post, topic_count) in enumerate(curr_user_posts_num_topics.items()):
        if topic_count == 0:
            print(f"Skipping post {i}... No topics found. ")
            continue

        if debug:
            print(f"\nExtracting topics for post {i}...")

        # Get the X # of topics per post per user
        extract_topics_prompt = ""
        if topic_count == 1:
            extract_topics_prompt += "I will provide you a Reddit comment, you will tell me what is the 1 topic that the comment talks about. "
            extract_topics_prompt += "Format your answer as a valid JSON list of the 1 topic as a string with no other output. "
            
        elif topic_count > 1:
            extract_topics_prompt += f"I will provide you a Reddit comment, you will tell me what are the {topic_count} topics that the comment talks about. "
            extract_topics_prompt += "Format your answer as a valid JSON list of the topics as strings with no other output. "
        extract_topics_prompt += "The comment is:\n"
        extract_topics_prompt += f'"""{post}"""'

        if debug:
            print("Prompt:", extract_topics_prompt)

        topics_output = gemini.AskGoogleGemini(extract_topics_prompt, force=force)
        
        try:
            topics_list = json.loads(topics_output)
        except Exception as e:
            print(f'{e}\nOutput:{topics_output}')

        if debug:
            print("List of topics:", topics_list)

        curr_user_posts_topics[post] = topics_list

    return curr_user_posts_topics
