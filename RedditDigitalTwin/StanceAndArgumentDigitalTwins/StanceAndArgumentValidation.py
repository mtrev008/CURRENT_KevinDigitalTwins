import polars as pl
import random
import json
import sys
sys.path.append('../../')
import GoogleGemini as gemini
from PipelineModules.GetNumTopics import GetNumTopicsPerPost
from PipelineModules.GetTopicsPerPost import GetTopicsPerPost
from PipelineModules.GetUserRealStance import ExtractRealUserStanceAboutTopics
from PipelineModules.InferUserStance import InferUserStanceOnHiddenTopic
from PipelineModules.InferUserArgument import InferUserArgumentOnHiddenTopic
import numpy as np
from utilities import Comment, Thread
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time

def GetUserPosts(df: pl.DataFrame, columnName: str) -> dict:
    """Gets all users' posts. Outputs a sorted dictionary."""

    user_posts_df = (
        df.group_by("author")
        .agg(pl.col(columnName).alias("posts"))
        .with_columns(pl.col("posts").list.len().alias("num_posts"))
        .sort("num_posts", descending=True)
    )

    user_posts = {}

    for row in user_posts_df.iter_rows(named=True):
        user = row["author"]
        posts = row["posts"]
        user_posts[user] = posts

    return user_posts

def main():
    # SET PARAMETERS
    num_users = 4 # Number of users to use for validation
    threshold_min_posts = 2 # Number of posts per user
    debugging = False # Set to False to turn off debugging print statements

    random.seed(2)

    # Init Google Gemini
    gemini.InitGoogleGemini()
    
    print(f"Starting validation script of {num_users} users with minimum {threshold_min_posts} # of posts...")

    timeOriginal = time.time() # DELETE THIS
    startTime = timeOriginal

    # Load data

    # Select a random {num_users} # of authors with at least {threshold_min_posts} # of posts
    # NOTE: This only returns the author names that have been selected for the sample
    exclude_posts = ["NaN", "nan", "Nan", "[deleted]", "[removed]", "[deleted by user]"]
    eligible_authors_DF = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet")
        .select(["author", "selftext"])
        .filter(pl.col("selftext").is_not_null())
        .filter(pl.col("selftext").is_in(exclude_posts).not_())
        .filter(pl.col("author") != "[deleted]")
        .group_by("author")
        .agg(pl.len().alias("num_posts")) # Count the # of posts for each author under column 'num_posts'
        .filter(pl.col("num_posts") >= threshold_min_posts)
        .sort("author") # Sort before sampling to ensure the same authors get chosen each run
        .collect(engine="streaming")
    )

    sampled_authors_names_DF = (
        eligible_authors_DF
        .sample(n=num_users, seed=2)
        .select("author")
    )

    author_names = sampled_authors_names_DF["author"].to_list()

    ################################################################################
    # DELETE THIS BLOCK WHEN THE ORIGINAL THREE USERS NO LONGER NEED TO BE INCLUDED
    # fixed_authors = ["limevince", "applebombers", "everybodyislying"]
    # additional_authors = (
    #     eligible_authors_DF
    #     .filter(pl.col("author").is_in(fixed_authors).not_())
    #     .sample(n=num_users - len(fixed_authors), seed=2)
    #     ["author"]
    #     .to_list()
    # )
    # author_names = fixed_authors + additional_authors
    ################################################################################

    if debugging:
        print("Selected user names:\n")
        print(author_names)

    # Get all posts by each selected user
    author_postsDF = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet")
        .filter(pl.col("selftext").is_not_null())
        .filter(pl.col("selftext").is_in(exclude_posts).not_())
        .filter(pl.col("author").is_in(author_names))
        .select(["author", "id", "title", "selftext"])
        .collect(engine="streaming")
    )

    print(f"Finished gathering author posts in {round(time.time()-startTime, 2)} seconds. ")

    if debugging:
        print("\nSelected users' posts:\n")
        print(author_postsDF)

    startTime = time.time()

    # Get all comments written by each selected user
    author_commentsDF = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_comments.parquet")
        .with_columns(
            pl.col("link_id").str.slice(3).alias("link_id"), # Ex) "t3_qwertyui" -> "qwertyui"
            pl.col("parent_id").str.slice(3).alias("parent_id") # Ex) "t1_qwertyui" -> "qwertyui"
        )
        .filter(pl.col("body").is_not_null())
        .filter(pl.col("body").is_in(exclude_posts).not_())
        .filter(pl.col("author").is_in(author_names))
        .select(["author", "id", "parent_id", "link_id", "body"])
        .collect(engine="streaming")
    )

    print(f"Finished gathering author comments in {round(time.time()-startTime, 2)} seconds. ")

    if debugging:
        print("\nSelected users' comments:\n")
        print(author_commentsDF)

    startTime = time.time()

    # Get all thread IDs that are relevant to the selected users. NOTE: This includes threads started by a selected user and threads where a selected user made a comment
    user_submission_thread_ids = author_postsDF["id"].unique().to_list()
    user_comment_thread_ids = author_commentsDF["link_id"].unique().to_list()

    thread_ids = list(set(user_submission_thread_ids + user_comment_thread_ids))

    # Set up threads where the user is a participant NOTE: this is different than the raw posts + comments per user
    # Get the original submissions for all relevant threads
    posts_df = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet")
        .filter(pl.col("id").is_in(thread_ids))
        .select(["id", "title", "selftext", "author"])
        .collect(engine="streaming")
    )

    if debugging:
        print('\nRelevant thread posts:\n')
        print(posts_df)

    # Set up Thread objects
    threads = {}

    for row in posts_df.iter_rows(named=True):
        thread = Thread(
            row["title"],
            row["selftext"],
            row["author"],
            row["id"]
        )

        threads[row["id"]] = thread

    # Load only comments that are ancestors of the selected users' comments.
    needed_parent_ids = set(author_commentsDF["parent_id"].drop_nulls().to_list())
    needed_parent_ids.difference_update(thread_ids)

    ancestor_rows = {}
    searched_ids = set()

    while needed_parent_ids:
        current_ids = needed_parent_ids - searched_ids

        if not current_ids:
            break

        searched_ids.update(current_ids)

        current_ancestorsDF = (
            pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_comments.parquet")
            .filter(pl.col("id").is_in(list(current_ids)))
            .filter(pl.col("body").is_not_null())
            .filter(pl.col("body").is_in(exclude_posts).not_())
            .select(["id", "parent_id", "link_id", "body", "author"])
            .collect(engine="streaming")
            .with_columns(
                pl.col("link_id").str.slice(3).alias("link_id"), # Ex) "t3_qwertyui" -> "qwertyui"
                pl.col("parent_id").str.slice(3).alias("parent_id") # Ex) "t1_qwertyui" -> "qwertyui"
            )
        )

        if current_ancestorsDF.is_empty():
            break

        next_parent_ids = set()

        for row in current_ancestorsDF.iter_rows(named=True):
            ancestor_rows[row["id"]] = row
            parent_id = row["parent_id"]

            if parent_id is not None and parent_id not in thread_ids and parent_id not in searched_ids:
                next_parent_ids.add(parent_id)

        needed_parent_ids = next_parent_ids

    # Set up dictionary where keys are users and values are content written by that user.
    user_posts = {} # the number of user-history items
    post_contexts = {} # the contextualized version of each history item

    # Add submissions written by each selected user
    for row in author_postsDF.iter_rows(named=True):
        post = row["title"] + ' ' + row["selftext"]

        user_posts.setdefault(row["author"], []).append(post)

        # No comment chain for a submission
        post_contexts[post] = {
            "thread": threads[row["id"]],
            "comment_chain": []
        }

    # Add comments written by each selected user
    for row in author_commentsDF.iter_rows(named=True):
        thread_id = row["link_id"]

        if thread_id not in threads:
            continue

        user_comment = Comment(
            row["body"],
            row["author"],
            row["id"],
            row["parent_id"],
            row["link_id"]
        )

        # Work backward from this user's comment to the root of its comment chain.
        comment_chain = [user_comment]
        parent_id = user_comment.parent_id
        visited_ids = {user_comment.id}

        while parent_id in ancestor_rows and parent_id not in visited_ids:
            ancestor = ancestor_rows[parent_id]
            parent_comment = Comment(
                ancestor["body"],
                ancestor["author"],
                ancestor["id"],
                ancestor["parent_id"],
                ancestor["link_id"]
            )

            comment_chain.insert(0, parent_comment)
            visited_ids.add(parent_comment.id)
            parent_id = parent_comment.parent_id

        # Raw user-authored comment
        post = user_comment.body

        user_posts.setdefault(row["author"], []).append(post)

        # Keep the thread and comment chain structurally separated
        post_contexts[post] = {
            "thread": threads[thread_id],
            "comment_chain": comment_chain
        }

    print(f"Finished setting up user histories in {round(time.time()-startTime, 2)} seconds. ")
    print(f"Total run time: {round(time.time()-timeOriginal, 2)} seconds.")


    #### Starting experiments
    
    print(f"# of selected users: {len(user_posts)}")

    all_user_post_topics_opinions = {} # Initialize empty final dict

    rounds = 0 # DELETE THIS
    tested_rounds = 0
    num_correct = 0 # DELETE THIS
    arguments_correct = 0 # DELETE THIS
    argument_rounds = 0 # DELETE THIS

    num_posts_list = []
    correct_list = []

    # Iterate through all users
    for user, history in user_posts.items(): # users_with_threshold_posts = {"user": ["post", "post", ...], ...}
        print(f"\nTesting user: '{user}'")
        print()

        tested_rounds += 1 # DELETE THIS

        # Raw posts + comments in one combined list
        history_raw = user_posts[user]

        # Get user's comments
        history_comments_only = [
            row["body"]
            for row in author_commentsDF.filter(pl.col("author") == user).iter_rows(named=True)
        ]

        # Context corresponding to the user's posts + comments
        history_context = {
            item: post_contexts[item]
            for item in user_posts[user]
        }

        # Get user's posts only
        history_posts_only = [
            row["title"] + ' ' + row["selftext"]
            for row in author_postsDF.filter(pl.col("author") == user).iter_rows(named=True)
        ]

        # Switch history depending on the desired input
        history = history_context
        # history = history_raw
        # history = history_posts_only

        # context_type = "posts only"
        context_type = "posts with comment chains"

        print(f"User '{user}' has {len(history)} history items...")

        ###### Step 1: Get the number of topics per posts (single integer)
        # curr_user_posts_num_topics = {"post": num_topics, ...}
        curr_user_posts_num_topics = GetNumTopicsPerPost(user, history, debug=False)


        print("**"*40)

        ###### Step 2: Get the actual topic(s) per post using the number of topics found in Step 1
        # curr_user_posts_topics = {"post": ["topic", "topic", ...], ...}
        curr_user_posts_topics = GetTopicsPerPost(curr_user_posts_num_topics, debug=False)

        print("**"*40)

        ###### Step 3: Get the positions of the user about every topic in every comment
        # TODO: DO NOT include topics here that the user is "not specified" towards.
        # TODO 2: ONLY call this function if the user has at least 2 unique topics that they either support or oppose.
        # posts_topics_positions = {"post": {"topic": position, ...}, ...}
        posts_topics_positions = ExtractRealUserStanceAboutTopics(user, history, curr_user_posts_topics, force=False, debug=False)
        
        print("**"*40)

        ###### Step 4: Ask another LLM to predict the user's viewpoint on a topic

        hidden_post, hidden_topic, hidden_opinion, predicted_opinion = InferUserStanceOnHiddenTopic(user, posts_topics_positions, context_type, history_context, force=False, debug=True) # Set force_random to True to randomize selection

        if hidden_post is None:
            continue

        print('\n' + '**'*40)

        if hidden_opinion == predicted_opinion: # DELETE THIS LATER
            num_correct +=1

        num_posts_list.append(len(history))

        if hidden_opinion == predicted_opinion:
            correct_list.append(1)
        else:
            correct_list.append(0)

        rounds += 1 # DELETE THIS

        ###### Step 5: Extract a user's real arguments, only include topics with support or oppose and that have a supporting argument
 
    
        ###### Step 6: Infer user's arguments

        # print("Inferring user arguments...")
        
        
        # hidden_post, hidden_topic, hidden_opinion, predicted_argument, argument_match = InferUserArgumentOnHiddenTopic(posts_topics_positions, debug=True)

        # if hidden_post is None:
        #     continue

        # argument_rounds += 1

        # # print("True:", hidden_opinion)
        # # print("Reasoning:", predicted_reasoning)
        # # print("Reasoning match:", reasoning_match)

        # if argument_match == "yes":
        #     print("\n" + "="*80)
        #     print("ARGUMENT MATCH FOUND")
        #     print("="*80)

        #     print("Topic:", hidden_topic)
        #     print("True stance:", hidden_opinion)
        #     print("Argument match:", argument_match)

        #     print("\nUSER REAL COMMENT:")
        #     print(hidden_post)

        #     print("\nLLM PREDICTED ARGUMENT:")
        #     print(predicted_argument)

        #     print("="*80 + "\n")
        #     arguments_correct += 1
        
        # continue # DELETE THIS

    print(f"Num correct: {num_correct}/{rounds}") # DELETE THIS
    quit() # DELETE THIS

    print(f"Num args correct: {arguments_correct}/{argument_rounds}")

    ######## Plot # of correct vs # of posts ########################
    X = np.array(num_posts_list).reshape(-1, 1)
    y = np.array(correct_list)

    model = LogisticRegression()
    model.fit(X, y)

    x_grid = np.linspace(min(num_posts_list), max(num_posts_list), 200).reshape(-1, 1)
    p_correct = model.predict_proba(x_grid)[:, 1]

    plt.scatter(num_posts_list, correct_list, alpha=0.2)
    plt.plot(x_grid, p_correct, linewidth=3)

    plt.xlabel("# Posts Per User")
    plt.ylabel("Correct Stance Prediction")
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.title("# Posts vs Stance Prediction Correctness")

    plt.show()
    
    quit()
        
    # TODO: check whether the opinion is correct (use both human annotation and standard metric)


    # Filter down users to users with > 1 topic
    # TODO: fix this to filter users not posts
    posts_with_at_least_one_topic = {}

    for post, topic_count in curr_user_posts_topics.items():
        if topic_count >= 1:
            posts_with_at_least_one_topic[post] = topic_count

    print("# of posts with at least one topic:", len(posts_with_at_least_one_topic.keys()))

    
if __name__ == '__main__':
    main()
