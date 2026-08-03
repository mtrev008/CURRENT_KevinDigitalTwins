import pandas as pd
import random
import json
import sys
sys.path.append('../')
import GoogleGemini as gemini

def PreprocessDF(df: pd.DataFrame, columnName: str) -> pd.DataFrame:
    # Drop rows with no text
    df = df.dropna(subset=[columnName])

    # Convert dataframe column to string
    df[columnName] = df[columnName].astype(str)

    df = df[df['author']!='[deleted]']

    df = df[df[columnName]!= 'NaN']
    df = df[(df[columnName] != 'nan') & (df[columnName] != '[deleted]') & (df[columnName] != '[removed]') & (df[columnName] != '[deleted by user]')]
    
    df = df.reset_index(drop=True)

    return df


def GetUserPosts(df:pd.DataFrame, columnName) -> dict:
    """Gets all users' posts. Outputs a sorted dictionary. """

    df = df.copy()

    user_posts = {}
    for i, row in df.iterrows():
        user = row['author']
        post = row[columnName]

        if user not in user_posts:
            user_posts[user] = []

        user_posts[user].append(post)

    # Sort dictionary by number of posts
    user_posts = dict(sorted(user_posts.items(), key=lambda item: len(item[1]), reverse=True))

    # Top 5 users
    # for user, posts in list(user_posts.items())[:5]:
    #     print(f"User: {user}, # of posts: {len(posts)}")

    return user_posts

def KeywordSearch(posts: dict, keyword: str) -> dict:
    """Finds posts that contain a keyword. """
    keyword = keyword.lower()

    posts_with_keyword = {}
    posts_without_keyword = {}

    for key, val in posts.items():
        for post in val:
            if keyword in post.lower():
                if key in posts_with_keyword:
                    posts_with_keyword[key].append(post)
                else:
                    posts_with_keyword[key] = []
                    posts_with_keyword[key].append(post)
            elif keyword not in post.lower():
                if key in posts_without_keyword:
                    posts_without_keyword[key].append(post)
                else:
                    posts_without_keyword[key] = []
                    posts_without_keyword[key].append(post)
                
        
    return posts_with_keyword, posts_without_keyword


def main():
    # SET USER PARAMS
    threshold = 5 # Number of posts per user
    pre = False
    random.seed(22)

    # Init Google Gemini
    gemini.InitGoogleGemini()

    if pre:
        print("Starting script...")
        df = pd.read_csv('InputData/PoliticalDiscussion_comments.csv')
        print("Preprocessing data...")
        df_pre = PreprocessDF(df, 'body')
        print("Getting user posts...")
        user_posts = GetUserPosts(df_pre, 'body')

        # Output user_posts to save time in experiments (run one time)
        with open("InputUserComments.json", "w") as f:
            json.dump(user_posts, f)
        quit()
    #################################################################
    with open("InputUserComments.json", "r") as f:
        user_posts = json.load(f)

    print(f"# of total users: {len(user_posts)}")
    users_5_posts = {}

    for key, val in user_posts.items():
        if len(val) >= 5:
            users_5_posts[key] = val
    
    print(f"# of users with >= 5 posts: {len(users_5_posts)}")
    
    # To get first user in dictionary
    # first_user = next(iter(users_5_posts))
    # print(first_user)

    # # Get user's posts
    # first_user_posts = user_posts[first_user]
    # print(len(first_user_posts))

    # Posts with keywords, also save posts without keyword
    posts_with_keyword, posts_without_keyword = KeywordSearch(users_5_posts, "nuclear power") 
    print(f"# of users with 'nuclear': {len(posts_with_keyword)}/{len(users_5_posts)}")

    # Randomly sample users
    sample_users_keyword = random.sample(list(posts_with_keyword.items()), 25) # Change number of samples
    sample_users_keyword = dict(sample_users_keyword)
    print('Users for testing:', list(sample_users_keyword.keys()))

    sample_users_posts_without_keyword = list(sample_users_keyword.keys()) # Users and their posts without keyword

    sample_users_keyword_LIST_DELETE = list(sample_users_keyword.keys())

    # Columns for our dataframe
    output_users = []
    output_post_histories = []
    output_test_posts = []
    output_labels = []

    for user in sample_users_keyword.keys():
        print('Current user:', user)
        output_users.append(user)

        posts_test = sample_users_keyword[user] # Current user's posts that contain the keyword
        # test_post = random.sample(posts_keyword, 1) # Sample one post with keyword for testing
        output_test_posts.append(posts_test) # All users' test posts for output dataframe

        post_history = posts_without_keyword[user]
        # post_history_prompting = random.sample(posts_no_keyword, 5) # Takes a sample of the post history, currently we use all posts
        output_post_histories.append(post_history)

        print("len of post history:",len(post_history))
        print("len of test posts:", len(posts_test))

        prompt = "You will adopt the personality of an online Reddit user who made the following posts. "
        prompt += "You must select from one of four possible standpoints on Nuclear Power and briefly provide your reasoning for doing so in 1-2 sentences as the user. " # be sure to correct for the number of options
        prompt += "The possible standpoints are:\n"
        prompt += "Standpoint 1: Implement rapid expansion and investment of nuclear power.\n"
        prompt += "Standpoint 2: Maintain the current nuclear energy operations without change.\n"
        prompt += "Standpoint 3: Phase out existing nuclear plants and halt new construction.\n"
        prompt += "Standpoint 4: Neutral or uncertain - Prioritize researching improved implementations of nuclear power.\n"
        prompt += "Here are the user's Reddit posts: \n"
        for i, post in enumerate(post_history):
            prompt += f"Post {i+1}: {post}\n\n"
        prompt += 'Format your response in JSON format such as {"Standpoint 2": "your reasoning"}:\n\n'

        output = gemini.AskGoogleGemini(prompt)
        output_labels.append(output)
        # print('TEST POSTS:', posts_test)
        print('**'*40)
        print('TEST POSTS:', posts_test)
        print('DIGITAL TWIN OUTPUT:', output)

    # Output results to csv
    annotate_df = pd.DataFrame()
    annotate_df['user'] = output_users
    annotate_df['post_history'] = output_post_histories
    annotate_df['test_posts'] = output_test_posts
    annotate_df['LLM_output'] = output_labels

    annotate_df.to_csv('LLM_labels_25_posts.csv', index=False) 

if __name__ == '__main__':
    main()