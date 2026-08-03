import json
import pandas as pd
import os

def GetRedditDataframeFromJsonl(columns: list[str], filename: str, infolder: str) -> pd.DataFrame:
    # Check if .csv already exists in cache, return that file
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    cache_filename = filename[:filename.find('.')] + '.csv' # Filename, but instead of .json it is .csv
    filepath = folder + 'DataCache/' + cache_filename

    if(os.path.isfile(filepath)):
        return pd.read_csv(filepath)

    # If no cache present, read from original .jsonl file and save as .csv cache file
    
    # Initialize a dictionary to keep our data
    data_dict = {}
    for column_name in columns:
        data_dict[column_name] = []

    # Open the .jsonl data file
    with open(infolder + filename, 'r') as f:
        # Iterate over every single post/comment
        for i, line in enumerate(f):
            # TEMPORARY: Cut down the dataset by 80% # DELETE THIS
            if(i < 7000000): # DELETE THIS
                continue # DELETE THIS

            post = json.loads(line)

            # Extract the desired columns
            for column_name in columns:
                data_dict[column_name].append(post[column_name])
            
            # Output some debugging info
            if(i % 100000 == 0 and i != 0):
                print(f"Loading post/comment #{i}...")
    print('Finished loading posts/comments.')

    # Convert the data dictionary to dataframe and cache it as .csv then return it
    df = pd.DataFrame(data_dict)
    df.to_csv(filepath, index=False)
    return df

def GetRedditPostsDataframe(filename="r_PoliticalDiscussion_posts.jsonl", infolder="/media/sf_Shared_Folder/Temp/") -> pd.DataFrame:
    "Returns a Pandas DataFrame of all posts from the given reddit jsonl posts file"
    # Fetch only the desired columns from posts
    columns = [
        'author',
        'created_utc',
        'id',
        'num_comments',
        'over_18',
        'permalink',
        'score',
        'upvote_ratio',
        'selftext',
        'subreddit',
        'subreddit_id',
        'title',
    ]

    return GetRedditDataframeFromJsonl(columns, filename, infolder)

def GetRedditCommentsDataframe(filename="r_PoliticalDiscussion_comments.jsonl", infolder="/media/sf_Shared_Folder/Temp/") -> pd.DataFrame:
    "Returns a Pandas DataFrame of all comments from the given reddit jsonl comments file"
    # Fetch only the desired columns from comments
    columns = [
        'id',
        'author',
        'subreddit_id',
        'subreddit',
        'score',
        'created_utc',
        'parent_id',
        'link_id',
        'body',
    ]

    return GetRedditDataframeFromJsonl(columns, filename, infolder)

def main():
    "Example usage"

    # df = GetRedditPostsDataframe()
    df = GetRedditCommentsDataframe()
    print("Earliest data date:", pd.to_datetime(df['created_utc'], unit='s').min())
    print("Latest data date:", pd.to_datetime(df['created_utc'], unit='s').max())

if __name__ == '__main__':
    main()