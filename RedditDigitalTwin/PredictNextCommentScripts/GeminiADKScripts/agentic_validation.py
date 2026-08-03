import pandas as pd
import sys
sys.path.append("../")
import json
import os
import asyncio
import hashlib
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from utilities import F1_Score_Recall_Precision, get_majority_vote
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import GoogleGemini as gemini


def make_agents():
    agent_1 = LlmAgent(
        model="gemini-3-flash-preview",
        name="agent_1",
        instruction=(
            "Does Post 2 represent Post 1's broad conclusion?\n"
            "Respond with either True or False.\n"
            "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
            "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
            "Do not require Post 2 to match Post 1 exactly.\n"
            "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
            'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"True","rationale":"short explanation"}'
        ),
    )

    agent_2 = LlmAgent(
        model="gemini-3-flash-preview",
        name="agent_2",
        instruction=(
            "Does Post 2 represent Post 1's broad conclusion?\n"
            "Respond with either True or False.\n"
            "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
            "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
            "Do not require Post 2 to match Post 1 exactly.\n"
            "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
            "Review the other agent's reasoning carefully before deciding.\n"
            'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"True","rationale":"short explanation"}'
        ),
    )

    return agent_1, agent_2


async def run_agent_cached(runner, user_id, session_id, prompt, max_retries=5):
    folder = os.path.dirname(os.path.abspath(__file__)) + '/'

    response = ''
    hashedPrompt = str(hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8])
    filepath = folder + 'GoogleAgentCache/' + hashedPrompt

    os.makedirs(folder + 'GoogleAgentCache/', exist_ok=True)

    if(os.path.isfile(filepath)):
        with open(filepath, 'r') as f:
            response = f.read()

    if(response != ''):
        return response

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)]
    )

    for i in range(max_retries):
        try:
            response = ''

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message
            ):
                if event.content:
                    text = ''
                    for part in event.content.parts:
                        if part.text:
                            text += part.text
                    if(text.strip() != ''):
                        response = text.strip()

            with open(filepath, 'w') as f:
                f.write(response)

            return response

        except Exception as e:
            print("API call failed:")
            print(e)
            if i < max_retries - 1:
                print("waiting 30 seconds and retrying...")
                await asyncio.sleep(30)
            else:
                return 'QUOTA_EXCEEDED'


def parse_json_response(text, default_response):
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except:
        print("bad response:")
        print(text)
        print()
        return default_response


def build_first_prompt(post_1, post_2):
    prompt = "Does Post 2 represent Post 1's broad conclusion? "
    prompt += "Respond with either True or False.\n"
    prompt += "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
    prompt += "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
    prompt += "Do not require Post 2 to match Post 1 exactly.\n"
    prompt += "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
    prompt += 'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"True","rationale":"short explanation"}\n'
    prompt += "\nPost 1:\n" + post_1
    prompt += "\n\nPost 2:\n" + post_2
    return prompt


def build_review_prompt(post_1, post_2, other_label, other_rationale):
    prompt = "Does Post 2 represent Post 1's broad conclusion? "
    prompt += "Respond with either True or False.\n"
    prompt += "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
    prompt += "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
    prompt += "Do not require Post 2 to match Post 1 exactly.\n"
    prompt += "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
    prompt += "Another agent has already answered.\n"
    prompt += "Review their label and rationale, then decide whether you agree or disagree.\n"
    prompt += "If you disagree, explain why and give your own label.\n"
    prompt += 'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"True","rationale":"short explanation"}\n'
    prompt += "\nPost 1:\n" + post_1
    prompt += "\n\nPost 2:\n" + post_2
    prompt += "\n\nOther agent label:\n" + other_label
    prompt += "\n\nOther agent rationale:\n" + other_rationale
    return prompt


def label_to_int(label):
    if str(label).lower() == "true":
        return 1
    return 0


async def main():
    user_id = "user_1"
    max_rounds = 10

    gemini.InitGoogleGemini(free_tier=False)

    agent_1, agent_2 = make_agents()

    runner_1 = InMemoryRunner(agent=agent_1, app_name="agent_1_app")
    runner_2 = InMemoryRunner(agent=agent_2, app_name="agent_2_app")

    annotations_df = pd.read_csv("DigitalTwins_PostPairAnnotations - TrueBias.csv")
    df = get_majority_vote(annotations_df)

    final_labels = []
    debate_rows = []

    for i in range(len(df)):
        print("analyzing post:", i)

        post_1 = df["User Post"][i]
        post_2 = df["LLM Output"][i]

        print("User post:", post_1)
        print()
        print("LLM output:", post_2)
        print()

        session_id_1 = f"agent_1_session_{i}"
        session_id_2 = f"agent_2_session_{i}"

        await runner_1.session_service.create_session(
            app_name="agent_1_app",
            user_id=user_id,
            session_id=session_id_1
        )

        await runner_2.session_service.create_session(
            app_name="agent_2_app",
            user_id=user_id,
            session_id=session_id_2
        )

        current_agent = 1
        current_label = ""
        current_rationale = ""
        final_label = "False"
        agreed = False

        rounds_used = 1
        agent_1_flipped = 0
        agent_2_flipped = 0
        agent_1_labels = []
        agent_2_labels = []
        agent_1_rationale_len = 0
        agent_2_rationale_len = 0
        first_agent_1_label = ""

        first_prompt = build_first_prompt(post_1, post_2)

        first_text = await run_agent_cached(
            runner_1,
            user_id,
            session_id_1,
            first_prompt
        )
        if first_text == 'QUOTA_EXCEEDED':
            print("quota exceeded on agent 1")
            return

        first_response = parse_json_response(
            first_text,
            {"label": "False", "rationale": ""}
        )

        current_label = str(first_response["label"]).strip()
        current_rationale = str(first_response["rationale"]).strip()
        final_label = current_label

        first_agent_1_label = current_label
        agent_1_labels.append(current_label)
        agent_1_rationale_len = len(current_rationale.split())

        print("round 1 - agent 1")
        print("label:", current_label)
        print("rationale:", current_rationale)
        print()

        for round_num in range(1, max_rounds + 1):
            if current_agent == 1:
                review_prompt = build_review_prompt(
                    post_1,
                    post_2,
                    current_label,
                    current_rationale
                )

                review_text = await run_agent_cached(
                    runner_2,
                    user_id,
                    session_id_2,
                    review_prompt
                )
                if review_text == 'QUOTA_EXCEEDED':
                    print("quota exceeded on agent 2")
                    return

                review_response = parse_json_response(
                    review_text,
                    {"label": "False", "rationale": ""}
                )

                new_label = str(review_response["label"]).strip()
                new_rationale = str(review_response["rationale"]).strip()

                rounds_used = round_num + 1
                agent_2_labels.append(new_label)
                agent_2_rationale_len = len(new_rationale.split())

                if len(agent_2_labels) >= 2:
                    if agent_2_labels[-1].lower() != agent_2_labels[-2].lower():
                        agent_2_flipped = 1

                print("round", round_num, "- agent 2")
                print("label:", new_label)
                print("rationale:", new_rationale)
                print()

                if new_label.lower() == current_label.lower():
                    agreed = True
                    final_label = new_label
                    break

                current_label = new_label
                current_rationale = new_rationale
                final_label = new_label
                current_agent = 2

            else:
                review_prompt = build_review_prompt(
                    post_1,
                    post_2,
                    current_label,
                    current_rationale
                )

                review_text = await run_agent_cached(
                    runner_1,
                    user_id,
                    session_id_1,
                    review_prompt
                )
                if review_text == 'QUOTA_EXCEEDED':
                    print("quota exceeded on agent 1")
                    return

                review_response = parse_json_response(
                    review_text,
                    {"label": "False", "rationale": ""}
                )

                new_label = str(review_response["label"]).strip()
                new_rationale = str(review_response["rationale"]).strip()

                rounds_used = round_num + 1
                agent_1_labels.append(new_label)
                agent_1_rationale_len = len(new_rationale.split())

                if len(agent_1_labels) >= 2:
                    if agent_1_labels[-1].lower() != agent_1_labels[-2].lower():
                        agent_1_flipped = 1

                print("round", round_num, "- agent 1")
                print("label:", new_label)
                print("rationale:", new_rationale)
                print()

                if new_label.lower() == current_label.lower():
                    agreed = True
                    final_label = new_label
                    break

                current_label = new_label
                current_rationale = new_rationale
                final_label = new_label
                current_agent = 1

        print("agreed:", agreed)
        print("final label:", final_label)
        print()

        hit_max_rounds = int(agreed == False)
        total_flips = agent_1_flipped + agent_2_flipped

        debate_rows.append({
            "post_index": i,
            "agent1_round1_label": label_to_int(first_agent_1_label),
            "final_label": label_to_int(final_label),
            "agreed": int(agreed),
            "rounds_used": rounds_used,
            "agent1_flipped": agent_1_flipped,
            "agent2_flipped": agent_2_flipped,
            "total_flips": total_flips,
            "agent1_rationale_len": agent_1_rationale_len,
            "agent2_rationale_len": agent_2_rationale_len,
            "rationale_len_diff": agent_1_rationale_len - agent_2_rationale_len,
            "hit_max_rounds": hit_max_rounds,
            "ground_truth": df["majority_vote"][i]
        })

        final_labels.append(final_label)
        
    pred_labels = []
    for label in final_labels:
        pred_labels.append(label_to_int(label))

    true_labels = df["majority_vote"].to_list()

    F1_Score_Recall_Precision(true_labels, pred_labels)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    print("Cohen kappa score:", kappa)
    print(confusion_matrix(true_labels, pred_labels))

    debate_df = pd.DataFrame(debate_rows)
    print(debate_df.head())
    debate_df.to_csv("AgentResults/debate_features_df.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())