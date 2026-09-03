import pandas as pd
import json
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()

def ExtractRealUserArgumentsAboutTopics(key, val, curr_user_posts_topics, posts_topics_positions, force=False, debug=False) -> dict:
    """Get the argument of a user for one topic and stance at a time (-1 = oppose, 1 = support).
    Returns: posts_topics_arguments = {"post": {"topic": {"position": position, "argument": argument}}, ...}"""

    topic_posts = {}
    for post, topics in curr_user_posts_topics.items():
        for topic in topics:
            topic_posts.setdefault(topic, []).append(post)

    selected_post = next(iter(posts_topics_positions))
    selected_topic = next(iter(posts_topics_positions[selected_post]))
    selected_position = posts_topics_positions[selected_post][selected_topic]
    # Get all posts about the target topic
    posts = topic_posts[selected_topic]

    if debug:
        print("\nExtracting argument...")

    if selected_position == 1:
        stance = "support"
    elif selected_position == -1:
        stance = "oppose"

    user_argument_prompt = ""
    user_argument_prompt += "I will provide you all of a Reddit user's posts about a topic. "
    user_argument_prompt += f'The user explicitly {stance}s "{selected_topic}". '
    user_argument_prompt += "Based on the posts, tell me the arguments the user makes for this stance. "
    user_argument_prompt += 'Focus on the claim and supporting reason. '
    user_argument_prompt += 'Format your answer as a valid JSON list of the arguments as strings with no other output. '
    

    if len(posts) == 1:
        user_argument_prompt += "The post is:\n"
    elif len(posts) > 1:
        user_argument_prompt += "The posts are:\n"

    for post in posts:
        user_argument_prompt += f'"""{post}"""\n'
    if debug:
        print("Prompt:", user_argument_prompt)

    response = gemini.AskGoogleGemini(user_argument_prompt, force=force)

    if debug:
        print("Output:", response)

    max_retries = 3
    for retries in range(max_retries + 1):
        try:
            argument = json.loads(response)
            break
        except Exception as e:
            print(f"{e}\nOutput: {response}")
            if retries == max_retries:
                return {}
            print(f"\nRetry #{retries + 1}: Prompting Gemini again\n")
            response = gemini.AskGoogleGemini(user_argument_prompt, force=True)

    # argument = str(response).strip()

    if len(argument) > 0:
        return {selected_post: {selected_topic: {"position": selected_position, "argument": argument}}}

    return {}
