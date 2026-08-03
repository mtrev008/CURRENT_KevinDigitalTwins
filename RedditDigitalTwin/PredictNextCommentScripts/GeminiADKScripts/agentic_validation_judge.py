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
    true_agent = LlmAgent(
        model="gemini-3-flash-preview",
        name="true_agent",
        instruction=(
            "You must argue that the correct label is True.\n"
            "True means Post 2 represents the broad conclusion of Post 1.\n"
            "Focus on the broad conclusion, not wording differences.\n"
            'Return raw JSON only. Do not use markdown or code fences: {"label":"True","rationale":"short explanation"}'
        ),
    )

    false_agent = LlmAgent(
        model="gemini-3-flash-preview",
        name="false_agent",
        instruction=(
            "You must argue that the correct label is False.\n"
            "False means Post 2 does not represent the broad conclusion of Post 1.\n"
            "Focus on meaningful disagreement in the broad conclusion.\n"
            'Return raw JSON only. Do not use markdown or code fences: {"label":"True","rationale":"short explanation"}'
        ),
    )

    judge_agent = LlmAgent(
        model="gemini-3-flash-preview",
        name="judge_agent",
        instruction=(
            "You are the judge.\n"
            "Read both arguments and decide which side is more correct.\n"
            "Label True if Post 2 represents the broad conclusion of Post 1.\n"
            "Label False if it does not.\n"
            "Focus on the broad conclusion, not wording differences.\n"
            'Return raw JSON only. Do not use markdown or code fences: {"label":"True","rationale":"short explanation"}'
        ),
    )

    return true_agent, false_agent, judge_agent


async def run_agent_cached(runner, user_id, session_id, prompt, max_retries=5):
    folder = os.path.dirname(os.path.abspath(__file__)) + '/'  # Folder of this script

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
    try:
        return json.loads(text)
    except:
        print("bad response:")
        print(text)
        print()
        return default_response


def build_true_prompt(post_1, post_2):
    prompt = "Argue why the correct label is True.\n"
    prompt += "Does Post 2 represent Post 1's broad conclusion?\n"
    prompt += "Respond with either True or False.\n"
    prompt += "True means Post 2 represents the broad conclusion of Post 1.\n"
    prompt += "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
    prompt += "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, then the correct label is True.\n"
    prompt += "Do not require Post 2 to match Post 1 exactly.\n"
    prompt += "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
    prompt += "You must argue for why True is the better label in this case.\n"
    prompt += 'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"True","rationale":"short explanation"}\n'
    prompt += "\nPost 1:\n" + post_1
    prompt += "\n\nPost 2:\n" + post_2
    return prompt


def build_false_prompt(post_1, post_2):
    prompt = "Argue why the correct label is False.\n"
    prompt += "Does Post 2 represent Post 1's broad conclusion?\n"
    prompt += "Respond with either True or False.\n"
    prompt += "False means Post 2 does not represent Post 1's broad conclusion.\n"
    prompt += "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
    prompt += "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, then the correct label is True.\n"
    prompt += "Do not require Post 2 to match Post 1 exactly.\n"
    prompt += "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
    prompt += "You must argue for why False is the better label in this case.\n"
    prompt += "To justify False, explain why Post 2 does not support the same broad conclusion or main takeaway as Post 1.\n"
    prompt += 'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"False","rationale":"short explanation"}\n'
    prompt += "\nPost 1:\n" + post_1
    prompt += "\n\nPost 2:\n" + post_2
    return prompt


def build_judge_prompt(post_1, post_2, true_rationale, false_rationale):
    prompt = "Decide whether Post 2 represents Post 1's broad conclusion.\n"
    prompt += "Respond with either True or False.\n"
    prompt += "You are given one argument for True and one argument for False.\n"
    prompt += "Choose the better side.\n"
    prompt += "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
    prompt += "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
    prompt += "Do not require Post 2 to match Post 1 exactly.\n"
    prompt += "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
    prompt += 'Return raw JSON only. Do not use markdown or code fences. Use this format exactly: {"label":"True","rationale":"short explanation"}\n'
    prompt += "\nPost 1:\n" + post_1
    prompt += "\n\nPost 2:\n" + post_2
    prompt += "\n\nArgument for True:\n" + true_rationale
    prompt += "\n\nArgument for False:\n" + false_rationale
    return prompt


def label_to_int(label):
    if str(label).lower() == "true":
        return 1
    return 0


async def main():
    user_id = "user_1"

    gemini.InitGoogleGemini(free_tier=False)

    true_agent, false_agent, judge_agent = make_agents()

    true_runner = InMemoryRunner(agent=true_agent, app_name="true_app")
    false_runner = InMemoryRunner(agent=false_agent, app_name="false_app")
    judge_runner = InMemoryRunner(agent=judge_agent, app_name="judge_app")

    annotations_df = pd.read_csv("DigitalTwins_PostPairAnnotations - TrueBias.csv")

    df = get_majority_vote(annotations_df) # Add majority vote column

    final_labels = []
    judge_rows = []

    for i in range(len(df)):
        print("analyzing post:", i)

        post_1 = df["User Post"][i]
        post_2 = df["LLM Output"][i]

        print("User post:", post_1)
        print()
        print("LLM output:", post_2)

        true_session_id = f"true_session_{i}"
        false_session_id = f"false_session_{i}"
        judge_session_id = f"judge_session_{i}"

        await true_runner.session_service.create_session(
            app_name="true_app",
            user_id=user_id,
            session_id=true_session_id
        )

        await false_runner.session_service.create_session(
            app_name="false_app",
            user_id=user_id,
            session_id=false_session_id
        )

        await judge_runner.session_service.create_session(
            app_name="judge_app",
            user_id=user_id,
            session_id=judge_session_id
        )

        true_prompt = build_true_prompt(post_1, post_2)
        false_prompt = build_false_prompt(post_1, post_2)

        true_text = await run_agent_cached(
            true_runner,
            user_id,
            true_session_id,
            true_prompt
        )
        if true_text == 'QUOTA_EXCEEDED':
            print("quota exceeded on true agent")
            return

        false_text = await run_agent_cached(
            false_runner,
            user_id,
            false_session_id,
            false_prompt
        )
        if false_text == 'QUOTA_EXCEEDED':
            print("quota exceeded on false agent")
            return

        true_response = parse_json_response(
            true_text,
            {"label": "True", "rationale": ""}
        )

        false_response = parse_json_response(
            false_text,
            {"label": "False", "rationale": ""}
        )

        true_rationale = str(true_response["rationale"]).strip()
        false_rationale = str(false_response["rationale"]).strip()

        judge_prompt = build_judge_prompt(
            post_1,
            post_2,
            true_rationale,
            false_rationale
        )

        judge_text = await run_agent_cached(
            judge_runner,
            user_id,
            judge_session_id,
            judge_prompt
        )
        if judge_text == 'QUOTA_EXCEEDED':
            print("quota exceeded on judge agent")
            return

        judge_response = parse_json_response(
            judge_text,
            {"label": "False", "rationale": ""}
        )

        final_label = str(judge_response["label"]).strip()

        final_labels.append(final_label)

        judge_rows.append({
            "post_index": i,
            "true_agent_label": label_to_int(true_response["label"]),
            "false_agent_label": label_to_int(false_response["label"]),
            "judge_label": label_to_int(final_label),
            "true_rationale_len": len(true_rationale.split()),
            "false_rationale_len": len(false_rationale.split()),
            "rationale_len_diff": len(true_rationale.split()) - len(false_rationale.split()),
            "ground_truth": df["majority_vote"][i]
        })

        print("true side:", true_rationale)
        print("false side:", false_rationale)
        print("final label:", final_label)
        print()

    pred_labels = []
    for label in final_labels:
        pred_labels.append(label_to_int(label))

    true_labels = df["majority_vote"].to_list()

    F1_Score_Recall_Precision(true_labels, pred_labels)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    print("Cohen kappa score:", kappa)
    print(confusion_matrix(true_labels, pred_labels))

    judge_df = pd.DataFrame(judge_rows)
    print(judge_df.head())
    judge_df.to_csv("AgentResults/judge_features_df.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())