import os
import polars as pl
import time

def main():
    "Convert the full .jsonl data file to a .parquet compressed format that takes instant time to run"
    # SET USER PARAMS HERE
    subreddit = 'PoliticalDiscussion'
    useCommentsFile = True # False = uses posts/submissions file, True uses comments file

    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    infolder = '/media/sf_Shared_Folder/Temp/' # Folder of original .jsonl data files
    if(useCommentsFile):
        infilename = f'r_{subreddit}_comments'
    else:
        infilename = f'r_{subreddit}_posts'

    startTime = time.time()
    
    # Manually set the data types for columns that cause errors
    # NOTE: This happens because polars guesses the column types based on the first 100 rows
    schema_overrides = {
        'author_flair_css_class': pl.String,
        'author_flair_text': pl.String,
        'distinguished': pl.String,
        'edited': pl.Int64,
    }

    # Verify that the cache file exists (.parquet is a compressed json format)
    cachefilename = folder + 'DataCache/' + infilename + '.parquet'
    if(not os.path.isfile(cachefilename)):
        print('Scanning and converting original large data file to faster cache format...')
        (
            pl.scan_ndjson(
                infolder + infilename + '.jsonl',
                schema_overrides=schema_overrides,
                infer_schema_length=None # Looks at the whole file to determine column type
            )
            .drop("media_embed")
            .drop("crosspost_parent_list")
            .sink_parquet(cachefilename)
        )
        print('Finished outputting cache file.')
    
    # # # # EXAMPLE USAGE # # # #
    # Prepare the query for finding the number of unique users in the dataset
    query = (
        pl.scan_parquet(cachefilename) # FROM
        .filter(pl.col('author') != '[deleted]') # WHERE: Filter out deleted users
        .select(
            pl.col('author').n_unique().alias("unique_users") # SELECT
        )
    )

    # Run the query which returns a DataFrame
    result_df = query.collect(engine="streaming") # Always need engine="streaming"
    print(result_df.schema) # DELETE THIS

    # Get the results, which in our case is a single row under a column called "unique_users"
    # NOTE: the .item() just gets the first result of the first row
    unique_users = result_df.item()

    if(useCommentsFile):
        print('Results using Comments file:')
    else:
        print('Results using Posts file:')
        
    print(f"# of unique users (excluding [deleted]):", unique_users)
    print('Total run time:', round(time.time() - startTime, 2), 'seconds.')

if __name__ == '__main__':
    main()