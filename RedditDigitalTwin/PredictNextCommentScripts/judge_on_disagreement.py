import pandas as pd
import sys
sys.path.append('../')
from open_source_llm import make_inference_call
import GoogleGemini as gemini
from utilities import get_cosine_similarity, get_jaccard_similarity, F1_Score_Recall_Precision, get_majority_vote, get_readability_score
from sklearn.metrics import cohen_kappa_score
import json

def setup_agent1(model, prompt):
    """Create first independent annotator"""
    current_prompt = prompt

    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"} or {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON.\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def setup_agent2(model, prompt):
    """Create second independent annotator"""
    current_prompt = prompt

    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"} or {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON.\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def setup_judge(model, prompt, agent1_label, agent1_reasoning, agent2_label, agent2_reasoning):
    """Create judge agent that decides final label only when independent annotators disagree"""
    current_prompt = prompt

    current_prompt += "\n\nTwo independent annotators disagreed.\n\n"

    current_prompt += "Annotator 1 response:\n"
    current_prompt += "Label: " + str(agent1_label) + "\n"
    current_prompt += "Explanation: " + str(agent1_reasoning) + "\n\n"

    current_prompt += "Annotator 2 response:\n"
    current_prompt += "Label: " + str(agent2_label) + "\n"
    current_prompt += "Explanation: " + str(agent2_reasoning) + "\n\n"

    current_prompt += "You are the final judge. Decide which label is better based on the original two texts and both annotators' reasoning.\n"
    current_prompt += "Focus on whether the LLM-generated and user-authored texts broadly make the same key points.\n"
    current_prompt += "Choose True if the two texts express the same core opinions and points. Choose False if they differ in their core opinions or key points.\n"
    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"} or {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON.\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def extract_label_and_reasoning(response):
    response = str(response).strip()

    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            response = response[start:end]

        parsed = json.loads(response)
        label = list(parsed.keys())[0]
        reasoning = list(parsed.values())[0]

        if str(label).lower() == "true":
            label = "True"
        else:
            label = "False"

        return label, reasoning

    except:
        response_lower = response.lower()
        if 'false' in response_lower:
            label = 'False'
        elif 'true' in response_lower:
            label = 'True'
        else:
            label = 'False'

        reasoning = response
        return label, reasoning

def main():
    # Init Google Gemini
    gemini.InitGoogleGemini(free_tier=False)

    debug=True

    annotations_df = pd.read_csv("AdditionalAnnotators_DigitalTwins_PostPairAnnotations - TrueBias.csv")

    groundtruth_df = annotations_df.copy()
    groundtruth_df['annotator_1'] = groundtruth_df['annotator_1'].astype(str).str.lower()
    groundtruth_df['annotator_1'] = groundtruth_df['annotator_1'].map({'true': 1, 'false': 0}).fillna(0)
    true_labels = groundtruth_df['annotator_1'].to_list()

    agent1_model = "gemini-3-flash-preview"
    agent2_model = "meta-llama/Llama-3.3-70B-Instruct"
    # judge_model = "deepseek-ai/DeepSeek-R1"
    judge_model = "gemini-3-flash-preview"

    predicted_labels = []

    agent1_labels = []
    agent1_reasonings = []
    agent2_labels = []
    agent2_reasonings = []
    judge_used = []
    judge_reasonings = []
    
    for i, post in enumerate(groundtruth_df['User Post']): 
        if debug:
            print("Analyzing post:", i)
        
        LLM_output = groundtruth_df["LLM Output"][i]

        prompt = "You are an expert annotator. In essence, are the LLM-generated and the user-authored texts broadly making the same key points? "
        # prompt += "Focus on the BROAD key points, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
        prompt += "Focus on the underlying meaning of the two texts, rather than exact wording. "
        prompt += "These texts do not need to use the same phrases or reasoning, but they should express the same core opinions and points.\n"
        prompt += "\nUser-authored text: \n"
        prompt += post
        prompt += "\nLLM-generated text: \n"
        prompt += LLM_output 

        if debug:
            print(prompt)
        
        response1 = setup_agent1(agent1_model, prompt)
        label1, reasoning1 = extract_label_and_reasoning(response1)

        response2 = setup_agent2(agent2_model, prompt)
        label2, reasoning2 = extract_label_and_reasoning(response2)

        used_judge = False
        judge_reasoning = ""

        if label1 == label2:
            final_label = label1
        else:
            used_judge = True

            judge_response = setup_judge(
                judge_model,
                prompt,
                label1,
                reasoning1,
                label2,
                reasoning2
            )

            final_label, judge_reasoning = extract_label_and_reasoning(judge_response)

        predicted_labels.append(final_label)

        agent1_labels.append(label1)
        agent1_reasonings.append(reasoning1)
        agent2_labels.append(label2)
        agent2_reasonings.append(reasoning2)
        judge_used.append(used_judge)
        judge_reasonings.append(judge_reasoning)

        if debug:
            print("Agent 1 Raw Response:", response1)
            print("Agent 1 Label:", label1)
            print("Agent 1 Reasoning:", reasoning1)
            print()

            print("Agent 2 Raw Response:", response2)
            print("Agent 2 Label:", label2)
            print("Agent 2 Reasoning:", reasoning2)
            print()

            if used_judge:
                print("Agents disagreed. Judge was used.")
                print("Judge Label:", final_label)
                print("Judge Reasoning:", judge_reasoning)
            else:
                print("Agents agreed. Judge was not used.")
                print("Final Label:", final_label)

            if groundtruth_df['annotator_1'][i] == 1:
                print("MY LABEL: True" )
            else:
                print("MY LABEL: False")
            print()

    for i, item in enumerate(predicted_labels):
        item = str(item).lower()
        predicted_labels[i] = item.replace(".", "")
        if item.lower() == 'true':
            predicted_labels[i] = 1
        else:
            predicted_labels[i] = 0

    F1_Score_Recall_Precision(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    print("Cohen kappa score:", kappa)
    print("Judge used:", sum(judge_used), "out of", len(judge_used))

    output_dict = {
        'agentic_predictions': predicted_labels,
        'agent1_label': agent1_labels,
        'agent1_reasoning': agent1_reasonings,
        'agent2_label': agent2_labels,
        'agent2_reasoning': agent2_reasonings,
        'judge_used': judge_used,
        'judge_reasoning': judge_reasonings
    }

    df = pd.DataFrame(output_dict)
    df.to_csv("AgentResults/Gemini_Llama_independent_judge_on_disagreement.csv", index=False)

if __name__=="__main__":
    main()