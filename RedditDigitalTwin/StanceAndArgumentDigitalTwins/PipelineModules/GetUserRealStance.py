import pandas as pd
import json
import random
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()

def ExtractRealUserStanceAboutTopics(key, val, curr_user_posts_topics, force=False, debug=False) -> dict:
    """Get the position of a user on one topic at a time (-1 = oppose, 1 = support, 0 = not specified).
    Returns: posts_topics_positions = {"post": {"topic": position, ...}, ...}"""

    topic_posts = {}
    for post, topics in curr_user_posts_topics.items():
        for topic in topics:
            topic_posts.setdefault(topic, []).append(post)

    # Shuffle all possible topics we can choose from
    topic_candidates = list(topic_posts)
    random.shuffle(topic_candidates)

    # Check each shuffled topic until the user has an explicit stance
    for i, topic in enumerate(topic_candidates):
        # Get every post with the topic
        posts = topic_posts[topic]
        if debug:
            print(f"\nExtracting opinion for topic candidate {i}...")

        user_position_prompt = ""
        user_position_prompt += "I will provide you all of a Reddit user's posts about a political topic. "
        user_position_prompt += "Based on all of the posts, tell me if the user explicitly supports, opposes, or "
        # Gets 85/100 correct
        # user_position_prompt += f"does not specify any stance towards {topic}. "
        # Gets 88/100 correct
        user_position_prompt += f'does not have an explicit stance on "{topic}". '
        # Gets 87/100 correct
        # user_position_prompt += f'does not express any stance on "{topic}". '
        user_position_prompt += 'Format your answer as either "support", "oppose", or "no stance" with no other output. '
        user_position_prompt += "The posts are:\n"
        for post in posts:
            user_position_prompt += f'"""{post}"""\n'
        if debug:
            print("Prompt:", user_position_prompt)

        response = gemini.AskGoogleGemini(user_position_prompt, force=force)

        if debug:
            print("Output:", response)

        if("support" in response.lower()):
            position = 1
        elif("oppose" in response.lower()):
            position = -1
        else:
            position = 0

        if position != 0:
            return {posts[0]: {topic: position}}

    return {}
