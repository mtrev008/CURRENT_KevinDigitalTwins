import json
import pandas as pd

def ExtractDataToCsv(infilepath: str, outfilepath: str, desired_columns: list):
    "Extract the raw reddit data into a .csv file with the given columns"
    print('Extracting data...')
    # Initialize a dictionary to keep our data
    data_dict = {}
    for column_name in desired_columns:
        data_dict[column_name] = []

    # Open the posts data file and extract relevant data
    with open(infilepath, 'r', encoding='utf-8') as f:
        # Each line in the file is a python dictionary
        for i, line in enumerate(f):
            if i%100 ==0:
                print(i)
            # Load each line as JSON (python) dictionary
            post = json.loads(line)

            # Extract the desired columns
            for column_name in desired_columns:
                data_dict[column_name].append(post[column_name])
    
    # Convert the data dictionary to dataframe and output as .csv
    df = pd.DataFrame(data_dict)
    df.to_csv(outfilepath, index=False)
    print(f'Done. Extracted {len(df)} rows.')

def main():
    "This script unpacks the relevant data from /InputData/ into a .csv file"

    # Extract submissions data
    """
    infilePath = 'InputData/PoliticalDiscussion_submissions'
    print(f'Converting: {infilePath}...')
    outfilepath = infilePath + '.csv'
    desired_columns = [
        'author',
        'created_utc',
        'id',
        'num_comments',
        'over_18',
        'permalink',
        'score',
        'selftext',
        'subreddit',
        'subreddit_id',
        'title',
    ]
    ExtractDataToCsv(infilePath, outfilepath, desired_columns)
    """
    # Extract comments data
    infilePath = 'InputData/PoliticalDiscussion_comments'
    print(f'Converting: {infilePath}...')
    outfilepath = infilePath + '.csv'
    desired_columns = [
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
    ExtractDataToCsv(infilePath, outfilepath, desired_columns)   

if __name__ == '__main__':
    main()