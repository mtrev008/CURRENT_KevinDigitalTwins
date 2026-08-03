import pandas as pd
import sys
sys.path.append('../')
from open_source_llm import make_inference_call
import GoogleGemini as gemini
from utilities import get_cosine_similarity, get_jaccard_similarity, F1_Score_Recall_Precision, get_majority_vote, get_readability_score
from sklearn.metrics import cohen_kappa_score
import json

def setup_agent1(model, prompt, opponent_label=None, opponent_reasoning=None, num_spaces=0):
    """Create first agent that chooses the initial label"""
    current_prompt = prompt

    if opponent_label is not None and opponent_reasoning is not None:
        current_prompt += "\n\nAnother annotator responded:\n"
        current_prompt += "Label: " + str(opponent_label) + "\n"
        current_prompt += "Explanation: " + str(opponent_reasoning) + "\n\n"
        current_prompt += "Given this other annotator's response, make your decision again.\n"

    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"} or {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON."
    if num_spaces > 1:
        current_prompt += " " * num_spaces + "\n" # Add spaces to make the prompts different per round
    else:
        current_prompt += "\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def setup_agent2(model, prompt, agent1_label, agent1_reasoning, num_spaces=0):
    """Create second agent that determines whether the first agent is correct"""
    current_prompt = prompt
    current_prompt += "\n\nAnother annotator responded:\n"
    current_prompt += "Label: " + str(agent1_label) + "\n"
    current_prompt += "Explanation: " + str(agent1_reasoning) + "\n\n"
    current_prompt += "Consider the opposite label and decide the correct label.\n"
    # current_prompt += "Do you agree or disagree with the other annotator?\n"
    # current_prompt += "Briefly critique their reasoning, then make your own decision based on the broad key points of the two texts.\n" # Test this next
    # current_prompt += "Make your own decision.\n" # works well
    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"} or {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON. "
    if num_spaces > 1:
        current_prompt += " " * num_spaces + "\n" # Add spaces to make the prompts different per round
    else:
        current_prompt += "\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=False)
    else:
        response = make_inference_call(current_prompt, model, force=False)

    return response

def setup_agent3(model, prompt, agent1_label, agent1_reasoning, agent2_label, agent2_reasoning):
    """Create third judge agent that decides final label after max rounds"""
    current_prompt = prompt

    current_prompt += "\n\nTwo annotators disagreed after multiple rounds.\n\n"

    current_prompt += "Annotator 1 final response:\n"
    current_prompt += "Label: " + str(agent1_label) + "\n"
    current_prompt += "Explanation: " + str(agent1_reasoning) + "\n\n"

    current_prompt += "Annotator 2 final response:\n"
    current_prompt += "Label: " + str(agent2_label) + "\n"
    current_prompt += "Explanation: " + str(agent2_reasoning) + "\n\n"

    current_prompt += "You are the final judge. Decide the correct label based on the original two texts and both annotators' reasoning.\n"
    current_prompt += "Focus on whether the LLM-generated and user-authored texts broadly make the same key points.\n"
    current_prompt += 'Output your response as valid JSON in this exact format: {"True": "brief explanation"} or {"False": "brief explanation"}.\n'
    current_prompt += "Do not output anything except the JSON.\n"

    if model == "gemini-3-flash-preview":
        response = gemini.AskGoogleGemini(current_prompt, force=True)
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

    max_rounds = 10
    debug=True

    annotations_df = pd.read_csv("AdditionalAnnotators_DigitalTwins_PostPairAnnotations - TrueBias.csv")

    groundtruth_df = annotations_df.copy()
    groundtruth_df['annotator_1'] = groundtruth_df['annotator_1'].astype(str).str.lower()
    groundtruth_df['annotator_1'] = groundtruth_df['annotator_1'].map({'true': 1, 'false': 0}).fillna(0)
    true_labels = groundtruth_df['annotator_1'].to_list()

    agent1_model = "gemini-3-flash-preview"
    agent2_model = "meta-llama/Llama-3.3-70B-Instruct"
    # agent2_model = "deepseek-ai/DeepSeek-R1"
    agent3_model = "gemini-3-flash-preview"

    models_predictions = {}

    predicted_labels = []
    rounds_used = []
    
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

        if debug:
            print("Agent 1 Raw Response:", response1)
            print("Agent 1 Label:", label1)
            print("Agent 1 Reasoning:", reasoning1)
            print()

        final_label = label1
        curr_rounds_used = 0

        last_agent2_label = None
        last_agent2_reasoning = None

        for round_num in range(1, max_rounds+1):
            response2 = setup_agent2(agent2_model, prompt, label1, reasoning1, num_spaces=curr_rounds_used)
            label2, reasoning2 = extract_label_and_reasoning(response2)

            last_agent2_label = label2
            last_agent2_reasoning = reasoning2

            curr_rounds_used += 1

            if debug:
                print("Round:", round_num)
                print("Agent 2 Raw Response:", response2)
                print("Agent 2 Label:", label2)
                print("Agent 2 Reasoning:", reasoning2)
                print()

            if label1 == label2:
                final_label = label1
                break

            response1 = setup_agent1(agent1_model, prompt, label2, reasoning2, num_spaces=curr_rounds_used)
            label1, reasoning1 = extract_label_and_reasoning(response1)

            if debug:
                print("Agent 1 Updated Raw Response:", response1)
                print("Agent 1 Updated Label:", label1)
                print("Agent 1 Updated Reasoning:", reasoning1)
                print()

            final_label = label1

        if curr_rounds_used == max_rounds and label1 != last_agent2_label:
            print("\n" + "="*80)
            print("POST INDEX:", i)
            print("ROUNDS USED:", curr_rounds_used)
            print("AGENT 1 FINAL LABEL:", label1)
            print("AGENT 2 FINAL LABEL:", last_agent2_label)
            print("FINAL LABEL:", final_label)
            print("USER POST:")
            print(post)
            print("\nLLM OUTPUT:")
            print(LLM_output)
            print("="*80 + "\n")
            response3 = setup_agent3(agent3_model, prompt, label1, reasoning1, last_agent2_label, last_agent2_reasoning)

            label3, reasoning3 = extract_label_and_reasoning(response3)
            final_label = label3

            if debug:
                print("Reached max rounds. Using Agent 3 as final judge.")
                print("Agent 3 Raw Response:", response3)
                print("Agent 3 Label:", label3)
                print("Agent 3 Reasoning:", reasoning3)
                print()

        predicted_labels.append(final_label)
        rounds_used.append(curr_rounds_used)

        if debug:
            print("Rounds Used:", curr_rounds_used)
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
    print("Average rounds used:", sum(rounds_used) / len(rounds_used))

    output_dict = {
        'agentic_predictions': predicted_labels,
        'rounds_used': rounds_used
    }

    df = pd.DataFrame(output_dict)
    df.to_csv("AgentResults/Gemini_Llama_agentic_debate.csv", index=False)

if __name__=="__main__":
    main()