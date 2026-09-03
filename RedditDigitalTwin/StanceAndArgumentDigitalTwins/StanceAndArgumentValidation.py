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
from PipelineModules.GetRealUserArguments import ExtractRealUserArgumentsAboutTopics
from PipelineModules.InferUserArgument import InferUserArgumentOnHiddenTopic
import numpy as np
from utilities import Comment, Thread
import time
import os

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
    num_users = 100 # Number of users to use for validation
    threshold_min_posts = 2 # Minimum number of posts per user
    threshold_min_comments = 2 # Minimum number of comments per user
    threshold_max_posts_and_comments = 50 # Users must have fewer than this many combined posts and comments
    debugging = False # Set to False to turn off debugging print statements

    random.seed(2)

    # Init Google Gemini
    gemini.InitGoogleGemini()

    print(f"Starting validation script of {num_users} users with minimum {threshold_min_posts} posts and {threshold_min_comments} comments...")

    timeOriginal = time.time() # DELETE THIS
    startTime = timeOriginal

    # Load data

    posts_path = "../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet"
    comments_path = "../FullDigitalTwin/DataCache/r_PoliticalDiscussion_comments.parquet"
    cache_folder = "../FullDigitalTwin/DataCache/ValidationSample/"
    os.makedirs(cache_folder, exist_ok=True)

    exclude_posts = ["NaN", "nan", "Nan", "[deleted]", "[removed]", "[deleted by user]"]

    # Select a random {num_users} # of authors with at least {threshold_min_posts} # of posts and {threshold_min_comments} # of comments
    selected_authors_cache = cache_folder + f"selected_authors_below_{threshold_max_posts_and_comments}_posts_and_comments_seed_2.parquet"

    eligible_post_counts = (
        pl.scan_parquet(posts_path)
        .select(["author", "selftext"])
        .filter(pl.col("selftext").is_not_null())
        .filter(pl.col("selftext").is_in(exclude_posts).not_())
        .filter(pl.col("author") != "[deleted]")
        .group_by("author")
        .agg(pl.len().alias("num_posts"))
        .filter(pl.col("num_posts") >= threshold_min_posts)
    )

    eligible_comment_counts = (
        pl.scan_parquet(comments_path)
        .select(["author", "body"])
        .filter(pl.col("body").is_not_null())
        .filter(pl.col("body").is_in(exclude_posts).not_())
        .filter(pl.col("author") != "[deleted]")
        .group_by("author")
        .agg(pl.len().alias("num_comments"))
        .filter(pl.col("num_comments") >= threshold_min_comments)
    )

    eligible_authors_DF = (
        eligible_post_counts
        .join(eligible_comment_counts, on="author", how="inner")
        .filter(pl.col("num_posts") + pl.col("num_comments") < threshold_max_posts_and_comments)
        .sort("author")
        .collect(engine="streaming")
    )

    if os.path.isfile(selected_authors_cache):
        sampled_authors_names_DF = pl.read_parquet(selected_authors_cache)

    else:

        sampled_authors_names_DF = (
            eligible_authors_DF
            .sample(n=num_users, seed=2)
            .select("author")
        )

        sampled_authors_names_DF.write_parquet(selected_authors_cache)

    author_names = sampled_authors_names_DF["author"].to_list()
    replacement_authors = (
        eligible_authors_DF
        .filter(pl.col("author").is_in(author_names).not_())
        .sample(fraction=1.0, shuffle=True, seed=2)
        ["author"]
        .to_list()
    )

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
    

    #### Starting experiments

    print(f"# of selected users: {len(author_names)}")

    all_user_post_topics_opinions = {} # Initialize empty final dict

    # Keep track of correct rounds
    rounds = {
        "posts with comment chains": 0,
        "posts only": 0,
        "posts and comments": 0
    } 
    
    tested_rounds = 0
    num_correct = {
        "posts with comment chains": 0,
        "posts only": 0,
        "posts and comments": 0
    } 

    arguments_correct = 0 # DELETE THIS
    argument_rounds = 0 # DELETE THIS

    num_posts_list = []
    correct_list = []
    argument_results = [] # DELETE THIS

    # Iterate through all users
    for i, user in enumerate(author_names):
        print(f"\nTesting user: '{user}' ({i}/{len(author_names)} users)...\n")

        tested_rounds += 1 # Keep track of rounds tested

        # See if user has already been checked
        user_cache_path = cache_folder + user + ".parquet"

        if os.path.isfile(user_cache_path):
            user_cache_DF = pl.read_parquet(user_cache_path)

        else:
            # If user does not exist in cache, retrieve user's post history
            startTime = time.time()

            # Get all posts by this selected user
            author_postsDF = (
                pl.scan_parquet(posts_path)
                .filter(pl.col("author") == user)
                .filter(pl.col("selftext").is_not_null())
                .filter(pl.col("selftext").is_in(exclude_posts).not_())
                .select(["author", "id", "title", "selftext"])
                .collect(engine="streaming")
            )

            print(f"Finished gathering author posts in {round(time.time()-startTime, 2)} seconds. ")

            if debugging:
                print("\nSelected user's posts:\n", author_postsDF)

            startTime = time.time()

            # Get all comments written by this selected user
            author_commentsDF = (
                pl.scan_parquet(comments_path)
                .filter(pl.col("author") == user)
                .with_columns(
                    pl.col("link_id").str.slice(3).alias("link_id"),
                    pl.col("parent_id").str.slice(3).alias("parent_id")
                )
                .filter(pl.col("body").is_not_null())
                .filter(pl.col("body").is_in(exclude_posts).not_())
                .select(["author", "id", "parent_id", "link_id", "body"])
                .collect(engine="streaming")
            )

            print(f"Finished gathering author comments in {round(time.time()-startTime, 2)} seconds. ")

            if debugging:
                print("\nSelected user's comments:\n", author_commentsDF)

            startTime = time.time()

            # Get all thread IDs that are relevant to this selected user. NOTE: This includes threads started by the selected user and threads where the selected user made a comment
            user_submission_thread_ids = author_postsDF["id"].unique().to_list()
            user_comment_thread_ids = author_commentsDF["link_id"].unique().to_list()

            thread_ids = list(set(user_submission_thread_ids + user_comment_thread_ids))

            # Set up threads where the user is a participant NOTE: this is different than the raw posts + comments per user
            # Get the original submissions for all relevant threads
            posts_df = (
                pl.scan_parquet(posts_path)
                .filter(pl.col("id").is_in(thread_ids))
                .select(["id", "title", "selftext", "author"])
                .collect(engine="streaming")
            )

            if debugging:
                print('\nRelevant thread posts:\n', posts_df)

            # Load only comments that are ancestors of the selected user's comments.
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
                    pl.scan_parquet(comments_path)
                    .filter(pl.col("id").is_in(list(current_ids)))
                    .filter(pl.col("body").is_not_null())
                    .filter(pl.col("body").is_in(exclude_posts).not_())
                    .select(["id", "parent_id", "link_id", "body", "author"])
                    .collect(engine="streaming")
                    .with_columns(
                        pl.col("link_id").str.slice(3).alias("link_id"),
                        pl.col("parent_id").str.slice(3).alias("parent_id")
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

            if ancestor_rows:
                ancestor_commentsDF = pl.DataFrame(list(ancestor_rows.values()))
            else:
                ancestor_commentsDF = pl.DataFrame(
                    schema={
                        "id": pl.String,
                        "parent_id": pl.String,
                        "link_id": pl.String,
                        "body": pl.String,
                        "author": pl.String
                    }
                )

            # Convert posts written by the selected user to the cache format
            author_posts_cache_DF = (
                author_postsDF
                .with_columns(
                    pl.lit("user_post").alias("record_type"),
                    pl.lit(None, dtype=pl.String).alias("parent_id"),
                    pl.col("id").alias("link_id"),
                    pl.col("selftext").alias("text")
                )
                .select(["record_type", "author", "id", "parent_id", "link_id", "title", "text"])
            )

            # Convert comments written by the selected user to the cache format
            author_comments_cache_DF = (
                author_commentsDF
                .with_columns(
                    pl.lit("user_comment").alias("record_type"),
                    pl.lit(None, dtype=pl.String).alias("title"),
                    pl.col("body").alias("text")
                )
                .select(["record_type", "author", "id", "parent_id", "link_id", "title", "text"])
            )

            # Convert relevant thread posts to the cache format
            thread_posts_cache_DF = (
                posts_df
                .with_columns(
                    pl.lit("thread_post").alias("record_type"),
                    pl.lit(None, dtype=pl.String).alias("parent_id"),
                    pl.col("id").alias("link_id"),
                    pl.col("selftext").alias("text")
                )
                .select(["record_type", "author", "id", "parent_id", "link_id", "title", "text"])
            )

            # Convert necessary ancestor comments to the cache format
            ancestor_comments_cache_DF = (
                ancestor_commentsDF
                .with_columns(
                    pl.lit("ancestor").alias("record_type"),
                    pl.lit(None, dtype=pl.String).alias("title"),
                    pl.col("body").alias("text")
                )
                .select(["record_type", "author", "id", "parent_id", "link_id", "title", "text"])
            )

            # Save all information necessary for this user
            user_cache_DF = pl.concat(
                [
                    author_posts_cache_DF,
                    author_comments_cache_DF,
                    thread_posts_cache_DF,
                    ancestor_comments_cache_DF
                ],
                how="vertical",
                rechunk=False
            )

            user_cache_DF.write_parquet(user_cache_path)

            print(f"Finished setting up user history in {round(time.time()-startTime, 2)} seconds. ")

        # Read this user's posts from the cache
        author_postsDF = (
            user_cache_DF
            .filter(pl.col("record_type") == "user_post")
            .select(["author", "id", "title", pl.col("text").alias("selftext")])
        )

        # Read this user's comments from the cache
        author_commentsDF = (
            user_cache_DF
            .filter(pl.col("record_type") == "user_comment")
            .select(["author", "id", "parent_id", "link_id", pl.col("text").alias("body")])
        )

        print(f"User '{user}' has {author_postsDF.height} posts and {author_commentsDF.height} comments.")

        # Read relevant thread posts from the cache
        posts_df = (
            user_cache_DF
            .filter(pl.col("record_type") == "thread_post")
            .select(["id", "title", pl.col("text").alias("selftext"), "author"])
        )

        # Read necessary ancestor comments from the cache
        ancestor_commentsDF = (
            user_cache_DF
            .filter(pl.col("record_type") == "ancestor")
            .select(["id", "parent_id", "link_id", pl.col("text").alias("body"), "author"])
        )

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

        # Set up Comment objects so shared ancestors do not need to be reconstructed for every comment chain
        comment_objects = {}

        for row in ancestor_commentsDF.iter_rows(named=True):
            comment_objects[row["id"]] = Comment(
                row["body"],
                row["author"],
                row["id"],
                row["parent_id"],
                row["link_id"]
            )

        for row in author_commentsDF.iter_rows(named=True):
            comment_objects[row["id"]] = Comment(
                row["body"],
                row["author"],
                row["id"],
                row["parent_id"],
                row["link_id"]
            )

        # Set up dictionary where keys are users and values are content written by that user.
        user_posts = {}
        post_contexts = {}

        # Add submissions written by the selected user
        for row in author_postsDF.iter_rows(named=True):
            post = row["title"] + ' ' + row["selftext"]

            user_posts.setdefault(row["author"], []).append(post)

            # No comment chain for a submission
            post_contexts[post] = {
                "thread": threads[row["id"]],
                "comment_chain": []
            }

        # Add comments written by the selected user
        for row in author_commentsDF.iter_rows(named=True):
            thread_id = row["link_id"]

            if thread_id not in threads:
                continue

            user_comment = comment_objects[row["id"]]

            # Work backward from this user's comment to the root of its comment chain.
            comment_chain = [user_comment]
            parent_id = user_comment.parent_id
            visited_ids = {user_comment.id}

            while parent_id in comment_objects and parent_id not in visited_ids:
                parent_comment = comment_objects[parent_id]

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

        # Raw posts + comments in one combined list
        history_raw = user_posts[user]

        # Get user's comments
        history_comments_only = [row["body"] for row in author_commentsDF.iter_rows(named=True)]

        # Context corresponding to the user's posts + comments (includes comment chains)
        history_context = {item: post_contexts[item] for item in user_posts[user]}

        # Get user's posts only
        history_posts_only = [row["title"] + ' ' + row["selftext"] for row in author_postsDF.iter_rows(named=True)]

        # SWITCH HISTORY DEPENDING ON THE DESIRED INPUT CONTEXT

        post_history_list = [history_context] #[history_posts_only, history_context, history_raw] #, history_comments_only  # TODO: Make this into a dictionary with history as key and context_type as val

        user_results = []

        for history in post_history_list:
            if history == history_posts_only:
                context_type = "posts only"
            elif history == history_raw:
                context_type = "posts and comments" 
            elif history == history_context:
                context_type = "posts with comment chains"

            # Topic extraction expects only the text of each user-authored history item
            history_text = list(history) if isinstance(history, dict) else history

            print(f"User '{user}' has {len(history)} history items...")
            print(f"Testing {context_type}...")

            ###### Step 1: Get the number of topics per posts (single integer)
            # curr_user_posts_num_topics = {"post": num_topics, ...}
            startTime = time.time()
            curr_user_posts_num_topics = GetNumTopicsPerPost(user, history_text, debug=False)
            print(f"\nFinished getting number of topics in {round(time.time() - startTime, 2)} seconds.")

            print("**"*40)

            ###### Step 2: Get the actual topic(s) per post using the number of topics found in Step 1
            # curr_user_posts_topics = {"post": ["topic", "topic", ...], ...}
            startTime = time.time()
            curr_user_posts_topics = GetTopicsPerPost(curr_user_posts_num_topics, debug=False)
            print(f"\nFinished getting topics in {round(time.time() - startTime, 2)} seconds.")

            print("**"*40)

            ###### Step 3: Get the positions of the user about every topic in every comment
            # TODO 2: ONLY call this function if the user has at least 2 unique topics that they either support or oppose.
            # posts_topics_positions = {"post": {"topic": position, ...}, ...}
            startTime = time.time()
            posts_topics_positions = ExtractRealUserStanceAboutTopics(user, history, curr_user_posts_topics, force=False, debug=False)
            print(f"\nFinished extracting stance in {round(time.time() - startTime, 2)}")

            if len(posts_topics_positions) == 0:
                if len(replacement_authors) == 0:
                    raise RuntimeError("No eligible replacement users are left.")

                replacement_user = replacement_authors.pop()
                author_names.append(replacement_user)
                print(f"Skipping user because no stance was found. Randomly selected replacement user '{replacement_user}'.")
                break

            print("**"*40)

            ###### Step 4: Ask another LLM to predict the user's viewpoint on a topic
            startTime = time.time()
            hidden_post, hidden_topic, hidden_stance, predicted_stance = InferUserStanceOnHiddenTopic(user, posts_topics_positions, curr_user_posts_topics, context_type, history_context, force=False, debug=False) # Set force_random to True to randomize selection
            print(f"\nFinished inferring stance in {round(time.time() - startTime, 2)}")

            # Check if there are no more users to sample from
            if hidden_post is None:
                if len(replacement_authors) == 0:
                    raise RuntimeError("No eligible replacement users are left.")

                replacement_user = replacement_authors.pop()
                author_names.append(replacement_user)
                print(f"Skipping user because no hidden prediction was found. Randomly selected replacement user '{replacement_user}'.")
                break

            print('\n' + '**'*40)

            user_results.append((context_type, len(history), hidden_stance, predicted_stance))

            ###### Step 5: Extract a user's real arguments, only include topics with support or oppose and that have a supporting argument
            # Skip argument evaluation for this context if we inferred stance incorrectly
            if hidden_stance.lower() != predicted_stance.lower():
                continue

            startTime = time.time()
            posts_topics_arguments = ExtractRealUserArgumentsAboutTopics(user, history, curr_user_posts_topics, posts_topics_positions, force=False, debug=False)
            print(f"\nFinished extracting arguments in {round(time.time() - startTime, 2)}")

            if not posts_topics_arguments:
                print("Skipping argument evaluation because no valid argument was extracted.")
                continue

            print(f'User posts about {hidden_topic}: \n{[post for post, topics in curr_user_posts_topics.items() if hidden_topic in topics]}')
            print(f"\nUser stance: {hidden_stance}")
            print(f'\nExtracted argument: {posts_topics_arguments[hidden_post][hidden_topic]["argument"]}')

            ###### Step 6: Infer user's arguments

            print("Inferring user arguments...")

            hidden_post, hidden_topic, hidden_opinion, predicted_argument = InferUserArgumentOnHiddenTopic(user, posts_topics_positions, curr_user_posts_topics, context_type, history_context, debug=False)

            print("\nPredicted argument:", predicted_argument)

            argument_results.append({"user": user, "target_topic": hidden_topic, "stance": hidden_opinion, "extracted_argument": json.dumps(posts_topics_arguments[hidden_post][hidden_topic]["argument"]), "predicted_argument": predicted_argument}) # DELETE THIS
            pl.DataFrame(argument_results).write_csv("argument_validation_results.csv") # DELETE THIS; currently outputs runs to CSV

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

        if len(user_results) != len(post_history_list):
            continue

        for context_type, history_length, hidden_stance, predicted_stance in user_results:
            # Count total number of times we were correct
            if hidden_stance.lower() == predicted_stance.lower(): # DELETE THIS LATER
                num_correct[context_type] +=1

            num_posts_list.append(history_length)

            if hidden_stance.lower() == predicted_stance.lower():
                correct_list.append(1)
            else:
                print("\nINCORRECT MATCH\n")
                print("\nHidden post:", hidden_post)
                print("Hidden topic:", hidden_topic)
                print("True position:", hidden_stance)
                print("Predicted position:", predicted_stance)
                correct_list.append(0)

            rounds[context_type] += 1 # Total number of rounds/users we ran

    for context_type in num_correct:
        print(f"{context_type}: {num_correct[context_type]}/{rounds[context_type]}") # DELETE THIS

    print(f"Num args correct: {arguments_correct}/{argument_rounds}")



    ######## Plot # of correct vs # of posts ########################
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

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



    # Filter down users to users with > 1 topic
    # TODO: fix this to filter users not posts
    posts_with_at_least_one_topic = {}

    for post, topic_count in curr_user_posts_topics.items():
        if topic_count >= 1:
            posts_with_at_least_one_topic[post] = topic_count

    print("# of posts with at least one topic:", len(posts_with_at_least_one_topic.keys()))


if __name__ == '__main__':
    main()
