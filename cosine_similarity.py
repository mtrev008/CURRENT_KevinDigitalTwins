import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import ast

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_aggregate(posts, llm_output):
    """Cosine similarity between aggregated user posts and LLM output"""

    sims = []

    for i, row in enumerate(posts):

        user_text = " ".join(row)

        user_emb = model.encode(user_text, convert_to_tensor=True)
        output_emb = model.encode(llm_output[i], convert_to_tensor=True)

        sim = util.cos_sim(user_emb, output_emb).item()

        sims.append(sim)

    return sum(sims)/len(sims)

def cosine_mean(posts, llm_output):
    """Average cosine similarity across all posts"""

    sims = []

    for i, row in enumerate(posts):

        output_emb = model.encode(llm_output[i], convert_to_tensor=True)

        post_sims = []

        for post in row:
            post_emb = model.encode(post, convert_to_tensor=True)
            sim = util.cos_sim(post_emb, output_emb).item()
            post_sims.append(sim)

        sims.append(sum(post_sims)/len(post_sims))

    return sum(sims)/len(sims)


def cosine_max(posts, llm_output):
    """Maximum cosine similarity between any post and the output"""

    sims = []

    for i, row in enumerate(posts):

        output_emb = model.encode(llm_output[i], convert_to_tensor=True)

        max_sim = -1

        for post in row:
            post_emb = model.encode(post, convert_to_tensor=True)
            sim = util.cos_sim(post_emb, output_emb).item()

            if sim > max_sim:
                max_sim = sim

        sims.append(max_sim)

    return sum(sims)/len(sims)


def main():
    df = pd.read_csv('LLM_labels_25_posts.csv')
    
    # df["LLM_output"] = df["LLM_output"].apply(json.loads)

     # convert string -> list
    df["post_history"] = df["post_history"].apply(ast.literal_eval) # Posts not about nuclear power
    df["test_posts"] = df["test_posts"].apply(ast.literal_eval) # Posts about nuclear power

    # Checking ALL posts:
    df["all_posts"] = df["test_posts"] + df["post_history"]

    # convert string -> dict
    df["LLM_output"] = df["LLM_output"].apply(ast.literal_eval)

    # extract reasoning text from {'Standpoint X': 'text'}
    df["LLM_output"] = df["LLM_output"].apply(lambda x: list(x.values())[0])

    print("Average Cosine Similarity\n")

    # print("Method 1 (Aggregate)")
    print("All Posts:", cosine_aggregate(df["all_posts"], df["LLM_output"]))
    # print("Posts about Nuclear:", cosine_aggregate(df["test_posts"], df["LLM_output"]))

    # print("\nMethod 2 (Mean)")
    print("All Posts:", cosine_mean(df["all_posts"], df["LLM_output"]))
    # print("Posts about Nuclear:", cosine_mean(df["test_posts"], df["LLM_output"]))

    # print("\nMethod 3 (Max)")
    print("All Posts:", cosine_max(df["all_posts"], df["LLM_output"]))
    # print("Posts about Nuclear:", cosine_max(df["test_posts"], df["LLM_output"]))


    



if __name__ == '__main__':
    main()