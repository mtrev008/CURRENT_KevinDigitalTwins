import pandas as pd
import json
import sys
sys.path.append('../../')
import utilities as ut
from utilities import Comment, Thread
import GoogleGemini as gemini
from preprocessing import preprocessing_data
import random

# Script for getting the next comment in a thread from LLM

def GetLongestThread(allThreads: list[Thread]) -> Thread:
    """Gets the thread with the most comments out of all threads"""
    longestThread = allThreads[0]
    for thread in allThreads:
        if(len(thread.comments) > len(longestThread.comments)):
            longestThread = thread
    return longestThread

def GetUserWithMostPostsInChain(commentChain: list[Comment]) -> str:
    """Returns the user with the most posts in the comment chain.
    Returns None if no user has at least 2 posts."""

    userCounts = {}

    for comment in commentChain:
        if(comment.user not in userCounts):
            userCounts[comment.user] = 0
        userCounts[comment.user] += 1

    user = None
    maxCount = 0
    for key, val in userCounts.items():
        if(val > maxCount):
            maxCount = val
            user = key

    if(user is None or maxCount < 2):
        return None

    return user


def main():
    print("Starting script...")

    # Init Google Gemini
    gemini.InitGoogleGemini()

    # Limit the number of rows if needed for testing
    numRows = 20000
    # pre_df_submissions = ut.get_submissions(numRows) # UNCOMMENT
    # print("# of submission loaded:", len(pre_df_submissions))
    pre_df_comments = ut.get_comments(chunk=True) #, numRows=600000) #numRows)
    print("# of comments loaded:", len(pre_df_comments))
    print("# of users:", pre_df_comments['author'].nunique()) # DELETE THIS
    print("First date:", pre_df_comments['created_utc'].min()) #DELETE THIS
    print("Last date:", pre_df_comments['created_utc'].max()) # DELETE THIS
    quit()

    # Preprocess dataframes
    df_submissions = preprocessing_data(pre_df_submissions, 'selftext') # Change back to not sample
    print("# of submissions AFTER preprocessing:", len(df_submissions))
    df_comments = preprocessing_data(pre_df_comments, 'body')
    print("# of comments AFTER preprocessing:", len(df_comments))

    # Initialize all threads
    allThreads = []
    for i, row in df_submissions.iterrows():
        thread = Thread(row['title'], row['selftext'], row['author'], row['id'])
        allThreads.append(thread)

    # Fill in the replies to each thread
    for i, row in df_comments.iterrows():
        comment = Comment(row['body'], row['author'], row['id'], row['parent_id'][3:], row['link_id'][3:])

        for j, thread in enumerate(allThreads):
            if(comment.link_id == thread.id):
                allThreads[j].comments.append(comment)

    cosine_similarities = []
    LLM_outputs = []
    user_posts = []
    result_df = pd.DataFrame()

    num_threads = 0 # DELETE LATER; use this to count the number of threads with more than 0 comments

    # Iterate through all threads 
    for i, thread in enumerate(allThreads): # Add [:X] to test sample of threads
        print(f"Thread #{i}...")
        if(len(thread.comments) == 0):
            print("Thread with no comments....")
            continue

        commentChain = thread.GetLongestCommentChain()
        user = GetUserWithMostPostsInChain(commentChain)
        if user == None:
            continue
        
        # Find the user's last post in the chain
        lastUserPostIndex = -1
        for j, comment in enumerate(commentChain):
            if(comment.user == user):
                lastUserPostIndex = j

        # Set up comment chain; Everything before the user's last post is the post history
        post_history = commentChain[:lastUserPostIndex]
        test_post = commentChain[lastUserPostIndex]

        ######### Use the following for adding new posts for annotations: #########
        # Check if the current test post exists in our annotated data, skip if the post exists
        current_posts_df = pd.read_csv("Annotations/PersonalCopy_DigitalTwins_PostPairAnnotations - TrueBias.csv") # Replace with current data annotations file
        
        temp_current_posts_df = current_posts_df[current_posts_df['User Post']==test_post.body]
        if len(temp_current_posts_df) >= 1: # Skip loop iteration if test_post exists in our annotated data already
            print("Post found in annotation data file. Skipping... ")
            continue
        ##########################################################################


    # DELETE THIS
    # Test with a random comment from the longest thread
    # longestTread = GetLongestThread(allThreads)
    # comment = longestTread.comments[61]
    # commentChain = longestTread.GetCommentChainAboveGivenComment(comment)
    # commentChain = commentChain[:12] # DELETE LATER (only for testing)

    # # Print the comment chain in chronological order:
    # # First the parent comment (AKA the submission)
    # print("user:", longestTread.user)
    # print("Post title:", longestTread.title)
    # print("Post body:", longestTread.body)
    # print('--'*40)
    # for i, comment in enumerate(commentChain):
    #     print(f"Comment #{i+1}:")
    #     print(f"User: {comment.user}")
    #     print()
    #     print(comment.body)
    #     print('--'*40)
    # quit()

        initial_post = thread.title + thread.body

        # Get the predicted response from Google Gemini
        prompt = "You are a Reddit user who has participated in the following political discussion thread. "
        # prompt += "Your task is to reply to the last comment in the thread as the user whose personality you adopted.\n"
        prompt += f"The title of the thread is: \"\"\"{thread.title}\"\"\".\n"
        prompt += f"The first post in the thread is: \"\"\"{thread.body}\"\"\".\n"
        prompt += "One of the comment chains in the thread that you participated in is as follows:\n"
        for i, com in enumerate(post_history):
            if com.user == user:
                prompt += f"You replied with this comment: \"\"\"{com.body}\"\"\".\n"
            else:
                prompt += f"A different user replied with this comment: \"\"\"{com.body}\"\"\".\n"
        prompt += "\n"
        prompt += "Your next reply in this comment chain will be:\n"

        print(prompt) # Comment out

        response = gemini.AskGoogleGemini(prompt)
        user_comment = test_post.body
        print("LLM output:")
        print(response)
        print('**'*40)
        print("User output")
        print(test_post.body)

        user_posts.append(user_comment)
        LLM_outputs.append(response)

        # Get cosine similarity
        # sentences = [response, user_comment]
        # similarity = ut.get_cosine_similarity(sentences)
        # print(f"Similarity: {similarity}")
        # cosine_similarities.append(similarity)

        if len(user_posts) >= 100:
            print('Saved 100 prompted comment chains.')
            result_df['User Post'] = user_posts
            result_df['LLM output'] = LLM_outputs
            # result_df['cosine similarity'] = cosine_similarities
            result_df.to_csv("Annotations/TEMP_truebias_posts_to_add.csv", index=False)
            return 0


    # Set up results dataframe
    result_df['User Post'] = user_posts
    result_df['LLM output'] = LLM_outputs
    # result_df['cosine similarity'] = cosine_similarities
    result_df.to_csv("Annotations/TEMP_truebias_posts_to_add.csv", index=False) # Output to csv
    print(len(result_df))

if __name__ == "__main__":
    main()