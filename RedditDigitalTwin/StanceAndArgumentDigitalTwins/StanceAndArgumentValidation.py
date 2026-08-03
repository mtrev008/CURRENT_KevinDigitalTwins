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
from utilities import PreprocessDF, Comment, Thread
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time

# DELETE THIS (OLD CODE)
# def PreprocessDF(df: pd.DataFrame, columnName: str) -> pd.DataFrame:
#     """Filter dataset to remove NaN content and deleted users. """

#     # Drop rows with no text
#     df = df.dropna(subset=[columnName])

#     # Convert dataframe column to string
#     df[columnName] = df[columnName].astype(str)

#     df = df[df['author']!='[deleted]']

#     df = df[(df[columnName]!= 'NaN') & (df[columnName] != 'nan') & (df[columnName] != '[deleted]') & (df[columnName] != '[removed]') & (df[columnName] != '[deleted by user]')]
    
#     df = df.reset_index(drop=True)

#     return df


# def GetUserPosts(df:pd.DataFrame, columnName) -> dict:
#     """Gets all users' posts. Outputs a sorted dictionary. """

#     df = df.copy()

#     user_posts = {}
#     for i, row in df.iterrows():
#         user = row['author']
#         post = row[columnName]

#         if user not in user_posts:
#             user_posts[user] = []

#         user_posts[user].append(post)

#     # Sort dictionary by number of posts
#     user_posts = dict(sorted(user_posts.items(), key=lambda item: len(item[1]), reverse=True))

#     # Top 5 users
#     # for user, posts in list(user_posts.items())[:5]:
#     #     print(f"User: {user}, # of posts: {len(posts)}")

#     return user_posts

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
    threshold = 2 # TODO: DELETE THIS, the line below does the same thing
    num_users = 3 # Number of users to use for validation
    threshold_min_posts = 2 # Number of posts per user

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
    sampled_authors_names_DF = (
        pl.read_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet", columns=["id", "author", "selftext"])
        .filter(pl.col("selftext").is_not_null())
        .filter(pl.col("selftext").is_in(exclude_posts).not_())
        .group_by("author")
        .agg(pl.len().alias("num_posts")) # Count the # of posts for each author under column 'num_posts'
        .filter(pl.col("num_posts") >= threshold_min_posts)
        .sort("author") # Sort before sampling to ensure the same authors get chosen each run
        .sample(n=num_users, seed=2)
        .select("author")
    )

    author_names = sampled_authors_names_DF["author"].to_list()

    endTime = time.time()
    print(f"Finished gathering sample in {round(endTime-startTime, 2)} seconds. ")
    startTime = endTime

    # Get all posts by each user
    author_postsDF = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet")
        .select(["author", "id", "title", "selftext"])
        .filter(pl.col("selftext").is_not_null())
        .filter(pl.col("selftext").is_in(exclude_posts).not_())
        .filter(pl.col("author").is_in(author_names))
        .collect(engine="streaming")
    )

    # print(author_postsDF.head())

    endTime = time.time()
    print(f"Finished gathering author posts in {round(endTime-startTime, 2)} seconds. ")
    startTime = endTime
    
    # Get all comments by each author
    author_commentsDF = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_comments.parquet")
        .select(["author", "id", "parent_id", "link_id", "body"])
        .with_columns(
            pl.col("link_id").str.slice(3).alias("link_id"), # Ex) "t3_qwertyui" -> "qwertyui"
            pl.col("parent_id").str.slice(3).alias("parent_id") # Ex) "t1_qwertyui" -> "qwertyui"
        )
        .filter(pl.col("body").is_not_null())
        .filter(pl.col("body").is_in(exclude_posts).not_())
        .filter(pl.col("author").is_in(author_names))
        .collect(engine="streaming")
    )

    # print(author_commentsDF.head())

    endTime = time.time()
    print(f"Finished gathering author comments in {round(endTime-startTime, 2)} seconds. ")
    startTime = endTime
    
    print(f"Total run time: {round(time.time() - timeOriginal, 2)} seconds.")
    quit()



    # FIX THIS IN A BIT -> use this to build threads, might not be needed tho
    # Get the sampled users' post IDs
    post_ids = (
        pl.read_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet")
        .filter(pl.col("author").is_in(sampled_authors_names_DF["author"]))
        .filter(pl.col("selftext").is_not_null())
        .filter(pl.col("selftext").is_in(exclude_posts).not_())
        .select("id") 
    )

    # Get comments data 
    # Fetch every comment corresponding to each post in post_ids
    # This is effectively a SQL join on "post.ids" = "comments.link_id"
    comments_df = (
        pl.scan_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_comments.parquet")
        .filter(pl.col("body").is_not_null())
        .filter(pl.col("body").is_in(exclude_posts).not_())
        .select(["id", "parent_id", "link_id", "body", "author"])
        .with_columns(
            pl.col("link_id").str.slice(3).alias("link_id"), # Ex) "t3_qwertyui" -> "qwertyui"
            pl.col("parent_id").str.slice(3).alias("parent_id") # Ex) "t1_qwertyui" -> "qwertyui"
        )
        .join(post_ids.lazy(), left_on="link_id", right_on="id", how="semi")
        .collect(engine="streaming")
    )

    # Get the actual post content for only those sampled post IDs
    posts_df = (
        pl.read_parquet("../FullDigitalTwin/DataCache/r_PoliticalDiscussion_posts.parquet")
        .filter(pl.col("id").is_in(post_ids["id"]))
        .select(["id", "title", "selftext", "author"])
    )

    print(posts_df.head())
    print(comments_df.head())

    print("# of posts:", posts_df.height)
    print("# of users in posts:", posts_df.select(pl.col("author").n_unique()).item())

    print("# of comments:", comments_df.height)
    print("# of users in comments:", comments_df.select(pl.col("author").n_unique()).item())

    combined_after = (pl.concat([posts_df.select("author"), comments_df.select("author")])
        .select(pl.col("author").n_unique())
        .item()
    )

    print("Combined # of unique users:", combined_after)
    print("Time ran:", time.time()- startTime)

    quit() # DELETE THIS

        
    # Set up threads

    threads = {}

    # Create thread objects from posts
    for row in posts_df.iter_rows(named=True):
        thread = Thread(
            row["title"],
            row["selftext"],
            row["author"],
            row["id"]
        )

        threads[row["id"]] = thread

    # Add comments to their matching thread
    for row in comments_df.iter_rows(named=True):
        thread_id = row["link_id"][3:]

        if thread_id not in threads:
            continue

        comment = Comment(
            row["body"],
            row["author"],
            row["id"],
            row["parent_id"][3:],
            row["link_id"][3:]
        )

        threads[thread_id].comments.append(comment)

    print(len(threads))
    quit() # DELETE THIS
    

    # Starting experiments

    # Set up dictionary where keys are users and values are their posts
    user_posts = preprocessed_comments_df.groupby("author")["body"].apply(list).to_dict()

    users_with_threshold_posts = {}

    # Get users with at least threshold number of posts
    for key, val in user_posts.items(): # key = user(str), val = posts (list[str])
        if len(val) >= threshold:
            users_with_threshold_posts[key] = val
    
    print(f"# of users with >= {threshold} posts: {len(users_with_threshold_posts)}")

    all_user_post_topics_opinions = {} # Initialize empty final dict

    rounds = 0 # DELETE THIS
    tested_rounds = 0
    num_correct = 0 # DELETE THIS
    arguments_correct = 0 # DELETE THIS
    argument_rounds = 0 # DELETE THIS

    num_posts_list = []
    correct_list = []

    # Iterate through all users
    for key, val in users_with_threshold_posts.items(): # users_with_threshold_posts = {"user": ["post", "post", ...], ...}
        if rounds > 25: # DELETE THIS
            break
        if key == '-Foxer': # DELETE THIS
            print("Skipping user:", key)
            break
        # if key != '---Sanguine---':
        #     continue

        # if rounds == 2: # DELETE THIS
        #     print(f"Skipped round {rounds}..")
        #     rounds += 1
        #     continue

        tested_rounds += 1 # DELETE THIS

        print(f"User '{key}' has {len(val)} posts...")

        ###### Step 1: Get the number of topics per posts (single integer)
        # curr_user_posts_num_topics = {"post": num_topics, ...}
        curr_user_posts_num_topics = GetNumTopicsPerPost(key, val, debug=False)

        # TODO: Remove posts with 0 topics

        # # # # TEMP # # # #
        # STEP 1: Get the list of (comment/# of topic) pairs for this user
        ##### UNCOMMENT LATER: ################
        # commentNumtopics = []
        # for post in val:
        #     num_topics = GetNumTopicsPerPost(post, debug=True)
        #     commentNumtopics.append((post, num_topics))
        #######################
        # userCommentNumtopics[key] = commentNumtopics # NOTE: Need to initialize this dict before loop
        # # # # TEMP # # # #

        print("**"*40)

        ###### Step 2: Get the actual topic(s) per post using the number of topics found in Step 1
        # curr_user_posts_topics = {"post": ["topic", "topic", ...], ...}
        curr_user_posts_topics = GetTopicsPerPost(curr_user_posts_num_topics, debug=False)

        print("**"*40)
        
        ###### Step 3: Get the positions of the user about every topic in every comment
        # TODO: DO NOT include topics here that the user is "not specified" towards.
        # TODO 2: ONLY call this function if the user has at least 2 unique topics that they either support or oppose.
        # posts_topics_positions = {"post": {"topic": position, ...}, ...}
        posts_topics_positions = ExtractRealUserStanceAboutTopics(key,val, curr_user_posts_topics, force=False, debug=False)
        
        print("**"*40)

        ###### Step 4: Ask another LLM to predict the user's viewpoint on a topic
        # TODO: ADD EACH USER'S POSTS FROM SUBMISSIONS FILE (in addition to comments)
        hidden_post, hidden_topic, hidden_opinion, predicted_opinion = InferUserStanceOnHiddenTopic(posts_topics_positions, debug=False) # Set force_random to True to randomize selection

        if hidden_post is None:
            continue

        # print("True:", hidden_opinion)
        # print("LLM:", predicted_opinion)

        print('\n' + '**'*40)

        if hidden_opinion == predicted_opinion: # DELETE THIS LATER
            num_correct +=1

        num_posts_list.append(len(val))

        if hidden_opinion == predicted_opinion:
            correct_list.append(1)
        else:
            correct_list.append(0)

        rounds += 1 # DELETE THIS

        ###### Step 5: Extract a user's real arguments, only include topics with support or oppose and that have a supporting argument
 
    
        ###### Step 6: Infer user's arguments

        print("Inferring user arguments...")
        
        
        hidden_post, hidden_topic, hidden_opinion, predicted_argument, argument_match = InferUserArgumentOnHiddenTopic(posts_topics_positions, debug=True)

        if hidden_post is None:
            continue

        argument_rounds += 1

        # print("True:", hidden_opinion)
        # print("Reasoning:", predicted_reasoning)
        # print("Reasoning match:", reasoning_match)

        if argument_match == "yes":
            print("\n" + "="*80)
            print("ARGUMENT MATCH FOUND")
            print("="*80)

            print("Topic:", hidden_topic)
            print("True stance:", hidden_opinion)
            print("Argument match:", argument_match)

            print("\nUSER REAL COMMENT:")
            print(hidden_post)

            print("\nLLM PREDICTED ARGUMENT:")
            print(predicted_argument)

            print("="*80 + "\n")
            arguments_correct += 1
        
        continue # DELETE THIS

    print(f"Num correct: {num_correct}/{rounds}") # DELETE THIS

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