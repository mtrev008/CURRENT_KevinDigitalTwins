import pandas as pd
import json
import random
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()

def InferUserStanceOnHiddenTopic(user, posts_topics_positions, context_type="max context", post_contexts=None, force=False, debug=False):
    """Sets up the prompt for getting a user's position about a randomly hidden topic.
    Param: posts_topics_positions = {"post": {"topic": position, ...}, ...}
    post_context = 'posts with comment chains' OR 'posts and comments' OR 'posts only'. """

    random.seed(3) # Originally 2 

    # Get posts that have at least one topic
    valid_posts = []

    for post, topics_positions in posts_topics_positions.items():
        if len(topics_positions) > 0:
            valid_posts.append(post)

    if len(valid_posts) == 0:
        print("Skipping user because there are no extracted topics.")
        return None, None, None, None

    # Randomly choose one post
    hidden_post = random.choice(valid_posts)

    # Randomly choose one topic from that post
    hidden_topic = random.choice(list(posts_topics_positions[hidden_post].keys()))

    # Get true hidden position
    hidden_position = posts_topics_positions[hidden_post][hidden_topic]

    # TODO: write posts and comments chains if statement, make the prompt setup like the predict_next_comment.py setup
    if context_type == "posts with comment chains":
        prompt = "You are a Reddit user who has participated in the following political discussion threads.\n"

        for post in posts_topics_positions.keys():
            if hidden_topic in posts_topics_positions[post]:
                continue

            context = post_contexts[post]
            thread = context["thread"]
            comment_chain = context["comment_chain"]

            prompt += f'\nThe title of the thread is: """{thread.title}""".\n'
            prompt += f'The first post in the thread is: """{thread.body}""".\n'

            for comment in comment_chain:
                if comment.body == hidden_post:
                    continue

                if comment.user == user:
                    prompt += f'You replied with this comment: """{comment.body}""".\n'
                else:
                    prompt += f'A different user replied with this comment: """{comment.body}""".\n'

    # If the post context is posts and comments as separate lists (no conversational context)
    elif context_type == "posts and comments":
        prompt = "You are a Reddit user who has previously written the following posts and comments:\n"

        for post in posts_topics_positions.keys():
            if hidden_topic in posts_topics_positions[post]:
                continue

            prompt += f'"""{post}"""\n'

    elif context_type == "posts only":
        prompt = "You are a Reddit user who has previously written the following posts:\n"

        for post in posts_topics_positions.keys():
            if hidden_topic in posts_topics_positions[post]:
                continue

            prompt += f'"""{post}"""\n'

    else:
        raise ValueError(f"Unknown context type: {context_type}")

    # prompt += f'What is your viewpoint on {hidden_topic}? Answer either "support", "oppose", or "neutral", with no other output.'
    # prompt += f'What is your stance towards {hidden_topic}? Answer either "In Favor", "Against", or "Neutral", with no other output.'
    # prompt += f'What is your stance towards {hidden_topic}? Answer either "support", "oppose", or "neutral", with no other output.' # This is the working version
    prompt += f'What is your stance on {hidden_topic}? Answer either "support", "oppose", or "no stance", with no other output.'

    if debug:
        print("Prompt:", prompt)

    predicted_position = gemini.AskGoogleGemini(prompt, force=force)
    predicted_position = str(predicted_position).strip().lower()

    if hidden_position == 1:
        hidden_position = "support"
    elif hidden_position == -1:
        hidden_position  = "oppose"
    else:
        hidden_position = "no stance"

    # if debug:
    print("Hidden post:", hidden_post)
    print("Hidden topic:", hidden_topic)
    print("True position:", hidden_position)
    print("Predicted position:", predicted_position)

    return hidden_post, hidden_topic, hidden_position, predicted_position


# Temporary testing function
def TEMP_InferUserPositionOnHiddenTopic(posts_topics_positions, randomize=True, force=False, debug=False):
    """Sets up the prompt for getting a user's position about a randomly hidden topic.
    Param: posts_topics_positions = {"post": {"topic": position, ...}, ...}"""

    # Exclude topics without a support or oppose position, and get rid of posts where the user is only "not specified" about its topics
    TEMP_posts_topics_positions = {}
    for post, topics_positions in posts_topics_positions.items(): # For every post of this user,
        TEMP_topics_positions = {}
        for topic, position in topics_positions.items(): # For every topic in this post
            # Save the topic and user's position as long as it's not "not specified" AKA 0
            if(position != 0):
                TEMP_topics_positions[topic] = position
        
        # If the post has any topics left that the user expressed a position, save them.
        # Do not bother saving posts where for every topic the user is "not specified" in their position.
        if(len(TEMP_topics_positions) > 0):
            TEMP_posts_topics_positions[post] = TEMP_topics_positions

    posts_topics_positions = TEMP_posts_topics_positions

    # If there are less than two posts or topics remaining, ignore this user
    unique_topics = []
    for post, topics_positions in posts_topics_positions.items():
        for topic in topics_positions:
            if(topic not in unique_topics):
                unique_topics.append(topic)
    
    if(len(posts_topics_positions) < 2 or len(unique_topics) < 2):
        return -1 # TODO: Move this entire check to main, for now returning -1 to cause obvious error


    # Now infer the digital twin's views on a randomly selected topic

    # Randomly select a topic to infer the opinion about
    topic = random.choice(unique_topics)

    # Get every comment that DOES NOT include this topic, and their topics and positions
    posts_topics_positions_with_topic = {} # {"post": {"topic": position, ...}, ...}
    posts_topics_positions_without_topic = {} # {"post": {"topic": position, ...}, ...}

    # For every post/comment of this user
    for post, topics_positions in posts_topics_positions.items():
        # Ignore the comment if the selected topic is mentioned in it, but save it for validation step later
        if(topic in topics_positions):
            posts_topics_positions_with_topic[post] = topics_positions

        # Otherwise, save it to feed to the LLM later
        else:
            posts_topics_positions_without_topic[post] = topics_positions



    # # TODO: Complete this prompt that excludes the user's specific views
    # prompt = ""
    # prompt += "I will provide you a Reddit user's views on certain topics "
    # prompt += "along with the comments from which they express those views. " # Can comment this line out for step 4. option b)
    # prompt += f"You will tell me if the user would support, oppose, or not have any clear position towards {topic}. "
    # prompt += 'Format your answer as either "support", "oppose", or "not specified" with no other output. '
    # prompt += "The user's views are as follows.\n\n"
    # for i, (post, topics_positions) in enumerate(posts_topics_positions_without_topic.items()):
    #     prompt += f'Comment #{i + 1}:\n'
    #     prompt += f'"""{post}""""\n'
    #     # NOTE: It might work better to not explicitly even provide this user's views on specific topics
    #     prompt += ""

    

    prompt = ""
    prompt += "I will provide you a Reddit user's views on certain topics "
    prompt += "along with the comments from which they express those views. " # Can comment this line out for step 4. option b)
    prompt += f"You will tell me if the user would support, oppose, or not have any clear position towards {topic}. "
    prompt += 'Format your answer as either "support", "oppose", or "not specified" with no other output. '
    prompt += "The user's views are as follows.\n\n"
    for i, (post, topics_positions) in enumerate(posts_topics_positions_without_topic.items()):
        prompt += f'Comment #{i + 1}:\n'
        prompt += f'"""{post}""""\n'
        # NOTE: It might work better to not explicitly even provide this user's views on specific topics
        views = []
        for topic, position in topics_positions.items():
            if(position == -1):
                views.append(f"opposes {topic}")
            else:
                views.append(f"supports {topic}")
        prompt += "In this comment the user " + ', '.join(views) + '.\n\n'


    if debug:
        print("Prompt:", prompt)

    response = gemini.AskGoogleGemini(prompt, force=force)
            
    if("support" in response.lower()):
        position = 1
    elif("oppose" in response.lower()):
        position = -1
    else:
        position = 0

    if debug:
        print(f"Predicted position on: {topic}: {position}")

    print(prompt)
    print('=='*40)
    print(response)
    quit()

    # TODO: Fix return values
    return topic, position
