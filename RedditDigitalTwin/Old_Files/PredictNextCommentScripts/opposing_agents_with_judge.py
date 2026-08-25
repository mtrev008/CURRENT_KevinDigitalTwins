import pandas as pd
import sys
sys.path.append('../')
from open_source_llm import make_inference_call
import GoogleGemini as gemini
from utilities import get_cosine_similarity, get_jaccard_similarity, F1_Score_Recall_Precision, get_majority_vote, get_readability_score
from sklearn.metrics import cohen_kappa_score
import json

def setup_true_agent(model, prompt):
    """Create agent that argues for the True label"""
    current_prompt = prompt

    current_prompt += "\n\nYou are Annotator 1. Your job is to make the strongest reasonable case for the label True.\n"
    current_prompt += "Assume the LLM-generated and user-authored texts broadly make the same key points, if that interpretation is reasonable.\n"
    current_prompt += "Focus on the underlying meaning of the two texts, rather than exact wording.\n"
    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON.\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def setup_false_agent(model, prompt):
    """Create agent that argues for the False label"""
    current_prompt = prompt

    current_prompt += "\n\nYou are Annotator 2. Your job is to make the strongest reasonable case for the label False.\n"
    current_prompt += "Assume the LLM-generated and user-authored texts do not broadly make the same key points, if that interpretation is reasonable.\n"
    current_prompt += "Focus on meaningful differences in core opinions, points, or underlying meaning.\n"
    current_prompt += 'Output your response as valid JSON in this exact format: {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON.\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def setup_judge(model, prompt, true_label, true_reasoning, false_label, false_reasoning):
    """Create judge agent that decides which opposing label is better"""
    current_prompt = prompt

    current_prompt += "\n\nTwo annotators gave opposing labels.\n\n"

    current_prompt += "Annotator 1 response:\n"
    current_prompt += "Label: " + str(true_label) + "\n"
    current_prompt += "Explanation: " + str(true_reasoning) + "\n\n"

    current_prompt += "Annotator 2 response:\n"
    current_prompt += "Label: " + str(false_label) + "\n"
    current_prompt += "Explanation: " + str(false_reasoning) + "\n\n"

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

    true_agent_model = "gemini-3-flash-preview"
    false_agent_model = "meta-llama/Llama-3.3-70B-Instruct"
    judge_model = "gemini-3-flash-preview"

    predicted_labels = []

    true_agent_labels = []
    true_agent_reasonings = []
    false_agent_labels = []
    false_agent_reasonings = []
    judge_reasonings = []
    
    for i, post in enumerate(groundtruth_df['User Post']): 
        if debug:
            print("Analyzing post:", i)
        
        LLM_output = groundtruth_df["LLM Output"][i]

        prompt = "You are an expert annotator. In essence, are the LLM-generated and the user-authored texts broadly making the same key points? "
        prompt += "Focus on the underlying meaning of the two texts, rather than exact wording. "
        prompt += "These texts do not need to use the same phrases or reasoning, but they should express the same core opinions and points.\n"
        prompt += "\nUser-authored text: \n"
        prompt += post
        prompt += "\nLLM-generated text: \n"
        prompt += LLM_output 

        if debug:
            print(prompt)
        
        true_response = setup_true_agent(true_agent_model, prompt)
        true_label, true_reasoning = extract_label_and_reasoning(true_response)

        false_response = setup_false_agent(false_agent_model, prompt)
        false_label, false_reasoning = extract_label_and_reasoning(false_response)

        judge_response = setup_judge(
            judge_model,
            prompt,
            true_label,
            true_reasoning,
            false_label,
            false_reasoning
        )

        final_label, judge_reasoning = extract_label_and_reasoning(judge_response)

        predicted_labels.append(final_label)

        true_agent_labels.append(true_label)
        true_agent_reasonings.append(true_reasoning)
        false_agent_labels.append(false_label)
        false_agent_reasonings.append(false_reasoning)
        judge_reasonings.append(judge_reasoning)

        if debug:
            print("True Agent Raw Response:", true_response)
            print("True Agent Label:", true_label)
            print("True Agent Reasoning:", true_reasoning)
            print()

            print("False Agent Raw Response:", false_response)
            print("False Agent Label:", false_label)
            print("False Agent Reasoning:", false_reasoning)
            print()

            print("Judge Raw Response:", judge_response)
            print("Final Label:", final_label)
            print("Judge Reasoning:", judge_reasoning)

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

    output_dict = {
        'agentic_predictions': predicted_labels,
        'true_agent_label': true_agent_labels,
        'true_agent_reasoning': true_agent_reasonings,
        'false_agent_label': false_agent_labels,
        'false_agent_reasoning': false_agent_reasonings,
        'judge_reasoning': judge_reasonings
    }

    df = pd.DataFrame(output_dict)
    df.to_csv("AgentResults/Gemini_Llama_opposing_agents_judge.csv", index=False)

if __name__=="__main__":
    main()