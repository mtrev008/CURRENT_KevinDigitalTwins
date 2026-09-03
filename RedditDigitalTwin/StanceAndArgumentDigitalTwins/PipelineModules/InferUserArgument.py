# PipelineModules/InferUserArgument.py
import pandas as pd
import json
import sys
sys.path.append('../')
import GoogleGemini as gemini

# Init Google Gemini
gemini.InitGoogleGemini()


def InferUserArgumentOnHiddenTopic(user, posts_topics_positions, curr_user_posts_topics, context_type="posts with comment chains", post_contexts=None, force=False, debug=False):
    """Predicts the argument a user would make for their real stance on a hidden topic, then
    checks whether the predicted argument matches the user's actual argument.

    posts_topics_positions = {"post": {"topic": position, ...}, ...}
    position: 1 = support, -1 = oppose, 0 = neutral """

    # Real stance extraction has already selected the single target topic
    hidden_post = next(iter(posts_topics_positions))
    hidden_topic = next(iter(posts_topics_positions[hidden_post]))

    # Get true hidden position
    hidden_position = posts_topics_positions[hidden_post][hidden_topic]

    if hidden_position == 1:
        hidden_position = "support"
    else:
        hidden_position = "oppose"

    # Ask digital twin for the argument GIVEN the real stance

    # If post context is posts with comment chains (conversational context); similar prompt setup as predict_next_comment.py
    if context_type == "posts with comment chains":
        prompt = "You are a Reddit user who has participated in the following political discussion threads.\n"
        threads = {}

        # Skip posts about target topic
        for post in curr_user_posts_topics.keys():
            if hidden_topic in curr_user_posts_topics[post]:
                continue

            context = post_contexts[post]
            thread = context["thread"]
            comment_chain = context["comment_chain"]

            if thread.id not in threads:
                threads[thread.id] = {"thread": thread, "comment_chain": {}}

            for comment in comment_chain:
                # Exclude any target topic comment written by the selected user,
                if (comment.user == user and hidden_topic in curr_user_posts_topics.get(comment.body, [])):
                    continue

                threads[thread.id]["comment_chain"][comment.id] = comment

        for context in threads.values():
            thread = context["thread"]
            comment_chain = context["comment_chain"].values()
            opener_text = thread.title + " " + thread.body

            # Check whether the title and/or post body are about the target topic
            user_target_topic_opener = (thread.user == user and hidden_topic in curr_user_posts_topics.get(opener_text, []))

            prompt += f'\nThe title of a thread is: """{thread.title}""".\n'

            # Check if the actual post body is empty
            if len(thread.body)==0:
                prompt += 'The first post in the thread is empty.\n'

            # Check if the user's post is about the target topic
            elif thread.user == user and not user_target_topic_opener:
                prompt += f'You posted the first post in the thread: """{thread.body}""".\n'
            else:
                prompt += f'A different user posted the first post in the thread: """{thread.body}""".\n'

            # Iterate over each comment, check if it was written by the user or not
            for comment in comment_chain:
                if comment.user == user:
                    prompt += f'You replied with this comment: """{comment.body}""".\n'
                else:
                    prompt += f'A different user replied with this comment: """{comment.body}""".\n'

    # If the post context is posts and comments as a list (no conversational context)
    elif context_type == "posts and comments":
        prompt = "You are a Reddit user who has previously written the following posts and comments:\n"

        for post in curr_user_posts_topics.keys():
            # Skip posts about the target topic
            if hidden_topic in curr_user_posts_topics[post]:
                continue

            prompt += f'"""{post}"""\n'

    # If the post context is posts only
    elif context_type == "posts only":
        prompt = "You are a Reddit user who has previously written the following posts:\n"

        # Skips posts about target topic
        for post in curr_user_posts_topics.keys():
            if hidden_topic in curr_user_posts_topics[post]:
                continue

            prompt += f'"""{post}"""\n'

    # Check if input is valid
    else:
        print(f"Unknown context type: {context_type}")

    prompt += f'Your stance towards {hidden_topic} is "{hidden_position}". '
    prompt += f'Based on your previous posts, what argument would you make to {hidden_position} {hidden_topic}? '
    prompt += 'Focus on the claim and supporting reason. '
    prompt += 'Answer with your argument and no other output. '
    # prompt += 'Answer as a valid JSON object with the key "argument" with no other output.'

    # if debug:
    #     print("Argument prediction prompt:", prompt)

    argument_response = gemini.AskGoogleGemini(prompt, force=force)

    try:
        predicted_argument = str(argument_response)
        # argument_json = json.loads(argument_response)
        # predicted_argument = argument_json["argument"]
    except Exception as e:
        print(f'{e}\nOutput:{argument_response}')
        predicted_argument = argument_response


    # DELETE THIS SECTION; it's bad slop
    # Ask another LLM if the predicted argument matches the real user's argument
    # argument_validation_prompt = f"""Topic: "{hidden_topic}"

    #     User stance: "{hidden_position}"

    #     Actual user comment:
    #     "{hidden_post}"

    #     Digital twin predicted argument:
    #     "{predicted_argument}"

    #     Are the user and the digital twin making the same argument to {hidden_position} {hidden_topic}?

    #     Reply as valid JSON:
    #     {{
    #         "match": "yes|partial|no",
    #         "explanation": "brief explanation"
    #     }}
    #     """

    # if debug:
    #     print("Argument validation prompt:", argument_validation_prompt)

    # argument_validation_response = gemini.AskGoogleGemini(argument_validation_prompt, force=force)

    # try:
    #     argument_validation_json = json.loads(argument_validation_response)
    #     argument_match = argument_validation_json["match"]
    #     argument_match_explanation = argument_validation_json["explanation"]
    # except Exception as e:
    #     print(f'{e}\nOutput:{argument_validation_response}')
    #     argument_match = argument_validation_response
    #     argument_match_explanation = argument_validation_response

    # argument_match = str(argument_match).strip().lower()
    #############################

    # if debug:
    #     print("Hidden post:", hidden_post)
    #     print("Hidden topic:", hidden_topic)
    #     print("True position:", hidden_position)
    #     print("Predicted argument:", predicted_argument)
    #     print("Argument match:", argument_match)
    #     print("Argument match explanation:", argument_match_explanation)

    return hidden_post, hidden_topic, hidden_position, predicted_argument
