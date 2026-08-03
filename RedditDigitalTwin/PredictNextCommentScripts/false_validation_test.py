import pandas as pd
import sys
sys.path.append('../')
import utilities as ut
import GoogleGemini as gemini
from scipy.stats import pointbiserialr
from open_source_llm import make_inference_call

def validation_test_with_false_intentions(df):
    """Two step process: 
    1) Prompt LLM to get topic from post
    2) Prompt LLM to come up with a comment for a political discussion subreddit """

    # Init Google Gemini
    gemini.InitGoogleGemini()

    posts = []
    topics = []
    LLM_output = []
    cosine_similarities = []

    # STEP 1: Get topic
    for post in df['user post']:
        prompt1 = "What is the broad topic that this Reddit comment is discussing? "
        prompt1 += "Output your response in 1-2 words. \n"
        prompt1 += "Comment: \n"
        prompt1 += post
        topic = gemini.AskGoogleGemini(prompt1, max_output_tokens=2048)
        print("Prompt1:", prompt1)
        print("Topic:", topic)
        print('***'*40)

        # STEP 2: Output comment 
        prompt2 = f"Generate an example Reddit comment on the topic of '{topic}' in 5 sentences or less and do not output anything else."
        output = gemini.AskGoogleGemini(prompt2)
        print("Prompt2:", prompt2)
        print("LLM Output:", output)

        posts.append(post)
        topics.append(topic)
        LLM_output.append(output)

        # Get cosine similarity
        sentences = [post, output]
        similarity = ut.get_cosine_similarity(sentences)
        print(f"Similarity: {similarity}")
        cosine_similarities.append(similarity)

    result_df = pd.DataFrame()
    result_df['post'] = posts
    result_df['topic'] = topics
    result_df['LLM_output'] = LLM_output
    result_df['cosine_similarity'] = cosine_similarities
    result_df.to_csv("Annotations/false_post_pair_validation.csv", index=False)


def main():
    # Run test
    df = pd.read_csv("true_post_pair_validation.csv")
    validation_test_with_false_intentions(df)
    quit()

    # Point-biseral correlation:
    # True intentions
    df_true = pd.read_csv("DigitalTwins_Annotations - TrueIntention_Labels.csv")
    df_true["annotation (True/False)"] = df_true["annotation (True/False)"].replace({True: 1, False: 0})
    x1 = df_true["annotation (True/False)"].to_list()
    y1 = df_true["cosine_similarity"].to_list()

    # False intentions
    df_false = pd.read_csv("DigitalTwins_Annotations - FalseIntention_Labels.csv")
    df_false["annotation (True/False)"] = df_false["annotation (True/False)"].replace({True: 1, False: 0})
    x2 = df_false["annotation (True/False)"].to_list()
    y2 = df_false["cosine_similarity"].to_list()
    
    # Get correlations for both combined
    x = x1 + x2
    y = y1 + y2
    # correlation, p_val = pointbiserialr(x, y)
    # print("Point-biseral correlation:", correlation)

    


if __name__ == "__main__":
    main()