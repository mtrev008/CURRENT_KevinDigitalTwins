# PipelineModules/InferUserArgument.py

import pandas as pd
import json
import random
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()


def CleanJsonResponse(response):
    response = response.strip()

    if response.startswith("```json"):
        response = response.replace("```json", "", 1).strip()

    if response.startswith("```"):
        response = response.replace("```", "", 1).strip()

    if response.endswith("```"):
        response = response[:-3].strip()

    return response


def InferUserArgumentOnHiddenTopic(posts_topics_positions, force=False, debug=False):
    """Predicts the argument a user would make for their real stance on a hidden topic, then 
    checks whether the predicted argument matches the user's actual argument.

    posts_topics_positions = {"post": {"topic": position, ...}, ...}
    position: 1 = support, -1 = oppose, 0 = neutral """

    random.seed(2)

    # Get posts that have at least one real non-neutral stance
    valid_posts = []

    for post, topics_positions in posts_topics_positions.items():
        real_stance_topics = []

        for topic, position in topics_positions.items():
            if position == 1 or position == -1:
                real_stance_topics.append(topic)

        if len(real_stance_topics) > 0:
            valid_posts.append(post)

    if len(valid_posts) == 0:
        print("Skipping user because there are no real non-neutral stances.")
        return None, None, None, None, None

    # Randomly choose one post
    hidden_post = random.choice(valid_posts)

    # Randomly choose one real non-neutral topic from that post
    valid_topics = []

    for topic, position in posts_topics_positions[hidden_post].items():
        if position == 1 or position == -1:
            valid_topics.append(topic)

    hidden_topic = random.choice(valid_topics)

    # Get true hidden position
    hidden_position = posts_topics_positions[hidden_post][hidden_topic]

    if hidden_position == 1:
        hidden_position = "support"
    elif hidden_position == -1:
        hidden_position = "oppose"
    else:
        hidden_position = "neutral"

    # Ask digital twin for the argument GIVEN the real stance
    prompt = ""
    prompt += "You are a Reddit user who has previously made the following comments:\n"

    for post in posts_topics_positions.keys():
        if hidden_topic in posts_topics_positions[post]:
            continue

        prompt += f'"""{post}"""\n\n'

    prompt += f'Your stance towards {hidden_topic} is "{hidden_position}". '
    prompt += f'Based on your previous comments, what argument would you make to {hidden_position} {hidden_topic}? '
    prompt += 'Focus on the claim and supporting reason. '
    prompt += 'Answer as a valid JSON object with the key "argument" with no other output.'

    # if debug:
    #     print("Argument prediction prompt:", prompt)

    argument_response = gemini.AskGoogleGemini(prompt, force=force)

    try:
        cleaned_argument_response = CleanJsonResponse(argument_response)
        argument_json = json.loads(cleaned_argument_response)
        predicted_argument = argument_json["argument"]
    except Exception as e:
        print(f'{e}\nOutput:{argument_response}')
        predicted_argument = argument_response

    # Ask another LLM if the predicted argument matches the real user's argument
    argument_validation_prompt = f"""Topic: "{hidden_topic}"

        User stance: "{hidden_position}"

        Actual user comment:
        "{hidden_post}"

        Digital twin predicted argument:
        "{predicted_argument}"

        Are the user and the digital twin making the same argument to {hidden_position} {hidden_topic}?

        Reply as valid JSON:
        {{
            "match": "yes|partial|no",
            "explanation": "brief explanation"
        }}
        """

    # if debug:
    #     print("Argument validation prompt:", argument_validation_prompt)

    argument_validation_response = gemini.AskGoogleGemini(argument_validation_prompt, force=force)

    try:
        cleaned_validation_response = CleanJsonResponse(argument_validation_response)
        argument_validation_json = json.loads(cleaned_validation_response)
        argument_match = argument_validation_json["match"]
        argument_match_explanation = argument_validation_json["explanation"]
    except Exception as e:
        print(f'{e}\nOutput:{argument_validation_response}')
        argument_match = argument_validation_response
        argument_match_explanation = argument_validation_response

    argument_match = str(argument_match).strip().lower()

    # if debug:
    #     print("Hidden post:", hidden_post)
    #     print("Hidden topic:", hidden_topic)
    #     print("True position:", hidden_position)
    #     print("Predicted argument:", predicted_argument)
    #     print("Argument match:", argument_match)
    #     print("Argument match explanation:", argument_match_explanation)

    return hidden_post, hidden_topic, hidden_position, predicted_argument, argument_match