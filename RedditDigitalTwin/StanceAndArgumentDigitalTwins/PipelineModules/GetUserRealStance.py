import pandas as pd
import json
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()

def ExtractRealUserStanceAboutTopics(key, val, curr_user_posts_topics, force=False, debug=False) -> dict:
    """Get the position of a user on one topic at a time (-1 = oppose, 1 = support, 0 = not specified).
    Returns: posts_topics_positions = {"post": {"topic": position, ...}, ...}"""

    posts_topics_positions = {} # {"post": {"topic": position, ...}, ...}

    for i, (post, topics) in enumerate(curr_user_posts_topics.items()): # curr_user_posts_topics = {"post": ["topic", "topic", ...], ...}
        if debug:
            print(f"\nExtracting opinions for post {i}...")

        topics_positions = {} # {"topic": position, ...} where (position = -1/0/1)

        for topic in topics:
            user_position_prompt = ""
            user_position_prompt += "I will provide you a Reddit post, you will tell me if the post text explicitly supports, opposes, or "
            user_position_prompt += f"does not specify any stance towards {topic}. "
            user_position_prompt += 'Format your answer as either "support", "oppose", or "no stance" with no other output. '
            # user_position_prompt += "I will provide you a Reddit comment, you will tell me if the comment text is explicitly in favor, against, or "
            # user_position_prompt += f"does not specify any stance towards {topic}. "
            # user_position_prompt += f"If there is a slight negative tone, classify it as 'Against'. "
            # # user_position_prompt += f"neutral {topic}. "
            # user_position_prompt += 'Format your answer as either "In Favor", "Against", or "Neutral" with no other output. '
            user_position_prompt += "The post is: "
            user_position_prompt += f'"""{post}"""'
            if debug:
                print("Prompt:", user_position_prompt)

            response = gemini.AskGoogleGemini(user_position_prompt, force=force)
            
            if("support" in response.lower()):
                position = 1
            elif("oppose" in response.lower()):
                position = -1
            else:
                position = 0

            if debug:
                print("Output:", position)

            if position == 0:
                continue

            topics_positions[topic] = position

        if len(topics_positions) > 0:
            posts_topics_positions[post] = topics_positions

    return posts_topics_positions
