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
            'Return raw JSON only. Do not use markdown or code fences. '
            'Use this format exactly: {"label":"True","rationale":"short explanation"}'
        ),
    )

    false_agent = LlmAgent(
        model="gemini-3-flash-preview",
        name="false_agent",
        instruction=(
            "You must argue that the correct label is False.\n"
            "False means Post 2 does not represent the broad conclusion of Post 1.\n"
            "Focus on meaningful disagreement in the broad conclusion.\n"
            'Return raw JSON only. Do not use markdown or code fences. '
            'Use this format exactly: {"label":"False","rationale":"short explanation"}'
        ),
    )

    judge_agent_1 = LlmAgent(
        model="gemini-3-flash-preview",
        name="judge_agent_1",
        instruction=(
            "You are judge 1.\n"
            "Read both arguments and decide which side is more correct.\n"
            "Label True if Post 2 represents the broad conclusion of Post 1.\n"
            "Label False if it does not.\n"
            "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
            "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
            "Do not require Post 2 to match Post 1 exactly.\n"
            "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
            'Return raw JSON only. Do not use markdown or code fences. '
            'Use this format exactly: {"label":"True","rationale":"short explanation"}'
        ),
    )

    judge_agent_2 = LlmAgent(
        model="gemini-3-flash-preview",
        name="judge_agent_2",
        instruction=(
            "You are judge 2.\n"
            "Read both arguments and decide which side is more correct.\n"
            "Label True if Post 2 represents the broad conclusion of Post 1.\n"
            "Label False if it does not.\n"
            "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
            "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
            "Do not require Post 2 to match Post 1 exactly.\n"
            "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
            'Return raw JSON only. Do not use markdown or code fences. '
            'Use this format exactly: {"label":"True","rationale":"short explanation"}'
        ),
    )

    judge_agent_3 = LlmAgent(
        model="gemini-3-flash-preview",
        name="judge_agent_3",
        instruction=(
            "You are judge 3.\n"
            "Read both arguments and decide which side is more correct.\n"
            "Label True if Post 2 represents the broad conclusion of Post 1.\n"
            "Label False if it does not.\n"
            "Focus on the BROAD conclusion, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
            "If both posts are broadly on the same side, express the same overall opinion, or support the same main takeaway, respond with True.\n"
            "Do not require Post 2 to match Post 1 exactly.\n"
            "Extra explanation, different examples, different emphasis, or stronger/weaker wording do NOT make them False by themselves.\n"
            'Return raw JSON only. Do not use markdown or code fences. '
            'Use this format exactly: {"label":"True","rationale":"short explanation"}'
        ),
    )

    return true_agent, false_agent, judge_agent_1, judge_agent_2, judge_agent_3


async def run_agent_cached(runner, user_id, session_id, prompt, cache_prefix="", max_retries=5):
    folder = os.path.dirname(os.path.abspath(__file__)) + '/'

    response = ''
    hashedPrompt = str(hashlib.md5((cache_prefix + prompt).encode('utf-8')).hexdigest()[:8])
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


def build_true_prompt(post_1, post_2):
    prompt = "Argue why the correct label is True.\n"
    prompt += "Does Post 2 represent Post 1's broad conclusion?\n"
    prompt += "Respond with either True or False.\n"
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


def majority_vote_labels(label_1, label_2, label_3):
    labels = [str(label_1).strip().lower(), str(label_2).strip().lower(), str(label_3).strip().lower()]

    true_count = labels.count("true")
    false_count = labels.count("false")

    if true_count >= 2:
        return "True"
    return "False"


async def main():
    user_id = "user_1"

    gemini.InitGoogleGemini(free_tier=False)

    true_agent, false_agent, judge_agent_1, judge_agent_2, judge_agent_3 = make_agents()

    true_runner = InMemoryRunner(agent=true_agent, app_name="true_app")
    false_runner = InMemoryRunner(agent=false_agent, app_name="false_app")
    judge_runner_1 = InMemoryRunner(agent=judge_agent_1, app_name="judge_app_1")
    judge_runner_2 = InMemoryRunner(agent=judge_agent_2, app_name="judge_app_2")
    judge_runner_3 = InMemoryRunner(agent=judge_agent_3, app_name="judge_app_3")

    annotations_df = pd.read_csv("DigitalTwins_PostPairAnnotations - TrueBias.csv")
    df = get_majority_vote(annotations_df)

    final_labels = []

    for i in range(len(df[:50])):
        print("analyzing post:", i)

        post_1 = df["User Post"][i]
        post_2 = df["LLM Output"][i]

        print("User post:", post_1)
        print()
        print("LLM output:", post_2)
        print()

        true_session_id = f"true_session_{i}"
        false_session_id = f"false_session_{i}"
        judge_session_id_1 = f"judge_session_1_{i}"
        judge_session_id_2 = f"judge_session_2_{i}"
        judge_session_id_3 = f"judge_session_3_{i}"

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

        await judge_runner_1.session_service.create_session(
            app_name="judge_app_1",
            user_id=user_id,
            session_id=judge_session_id_1
        )

        await judge_runner_2.session_service.create_session(
            app_name="judge_app_2",
            user_id=user_id,
            session_id=judge_session_id_2
        )

        await judge_runner_3.session_service.create_session(
            app_name="judge_app_3",
            user_id=user_id,
            session_id=judge_session_id_3
        )

        true_prompt = build_true_prompt(post_1, post_2)
        false_prompt = build_false_prompt(post_1, post_2)

        true_text = await run_agent_cached(
            true_runner,
            user_id,
            true_session_id,
            true_prompt,
            cache_prefix="true_agent_"
        )
        if true_text == 'QUOTA_EXCEEDED':
            print("quota exceeded on true agent")
            return

        false_text = await run_agent_cached(
            false_runner,
            user_id,
            false_session_id,
            false_prompt,
            cache_prefix="false_agent_"
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

        judge_text_1 = await run_agent_cached(
            judge_runner_1,
            user_id,
            judge_session_id_1,
            judge_prompt,
            cache_prefix="judge_1_"
        )
        if judge_text_1 == 'QUOTA_EXCEEDED':
            print("quota exceeded on judge agent 1")
            return

        judge_text_2 = await run_agent_cached(
            judge_runner_2,
            user_id,
            judge_session_id_2,
            judge_prompt,
            cache_prefix="judge_2_"
        )
        if judge_text_2 == 'QUOTA_EXCEEDED':
            print("quota exceeded on judge agent 2")
            return

        judge_text_3 = await run_agent_cached(
            judge_runner_3,
            user_id,
            judge_session_id_3,
            judge_prompt,
            cache_prefix="judge_3_"
        )
        if judge_text_3 == 'QUOTA_EXCEEDED':
            print("quota exceeded on judge agent 3")
            return

        judge_response_1 = parse_json_response(
            judge_text_1,
            {"label": "False", "rationale": ""}
        )

        judge_response_2 = parse_json_response(
            judge_text_2,
            {"label": "False", "rationale": ""}
        )

        judge_response_3 = parse_json_response(
            judge_text_3,
            {"label": "False", "rationale": ""}
        )

        judge_label_1 = str(judge_response_1["label"]).strip()
        judge_label_2 = str(judge_response_2["label"]).strip()
        judge_label_3 = str(judge_response_3["label"]).strip()

        final_label = majority_vote_labels(
            judge_label_1,
            judge_label_2,
            judge_label_3
        )

        final_labels.append(final_label)

        print("true side:", true_rationale)
        print("false side:", false_rationale)
        print("judge 1 label:", judge_label_1)
        print("judge 2 label:", judge_label_2)
        print("judge 3 label:", judge_label_3)
        print("final label:", final_label)
        print()

    pred_labels = []
    for label in final_labels:
        pred_labels.append(label_to_int(label))

    true_labels = df["majority_vote"].to_list()

    F1_Score_Recall_Precision(true_labels[:50], pred_labels)
    kappa = cohen_kappa_score(true_labels[:50], pred_labels)
    print("Cohen kappa score:", kappa)
    print(confusion_matrix(true_labels[:50], pred_labels))


if __name__ == "__main__":
    asyncio.run(main())