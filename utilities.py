import os
import pandas as pd
import polars as pl
import ast
# Uncomment this if needed
# from sentence_transformers import SentenceTransformer # Assigns numerical values to text to help with paraphrasing/finding meaning within a sentence (text -> vector)
from sklearn.metrics.pairwise import cosine_similarity # Uses the numerical values assigned to compare them  and guage if sentences have the same meaning!!
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import re
import textstat
import krippendorff

# Helper functions

# Setting up Thread and Comment classes
class Comment:
    def __init__(self, body, user, id, parent_id, link_id):
        self.body = body
        self.user = user
        self.id = id
        self.parent_id = parent_id
        self.link_id = link_id

    def toJSON(self):
        return {"body": self.body, "user": self.user, "id": self.id, "parent_id": self.parent_id, "link_id": self.link_id}
    

class Thread:
    def __init__(self, title, body, user, id):
        self.title = title
        self.body = body
        self.user = user
        self.id = id
        self.comments: list[Comment] = [] # Initialize this thread with an empty comment reply list
    
    def toJSON(self):
        return {"title": self.title, "body": self.body, "user": self.user, "id": self.id}
    
    def GetCommentChainAboveGivenComment(self, comment: Comment) -> list[Comment]:
        "Get the comment chain above the given comment"
        commentChain = []
        commentChain.append(comment)

        valuesChanged = True
        while(valuesChanged):
            # If we do not find a parent comment we will exit the loop
            valuesChanged = False

            for com in self.comments:
                # If we found a parent comment to the first comment
                if(com.id == commentChain[0].parent_id):
                    commentChain.insert(0, com) # Insert comment at beginning of list
                    valuesChanged = True # Indicates that we will loop again
                    break

                # If we found a child comment to the last comment
                # TODO: DELETE THIS since this elif section just goes down one possible path of the comment tree from this point
                # elif(com.parent_id == commentChain[-1].id):
                #     commentChain.append(com)
                #     valuesChanged = True
                #     break
        
        return commentChain


    # def GetAllLeafNodes(self) -> list[Comment]:
    #     "Returns all leaf node comments in this thread"
    
    def GetLongestCommentChain(self) -> list[Comment]:
        "Get the longest comment chain in this thread"

        longestCommentChain = []

        for comment in self.comments:
            isLeaf = True

            for otherComment in self.comments:
                if(otherComment.parent_id == comment.id):
                    isLeaf = False
                    break

            if(isLeaf):
                commentChain = self.GetCommentChainAboveGivenComment(comment)

                if(len(commentChain) > len(longestCommentChain)):
                    longestCommentChain = commentChain

        return longestCommentChain



def get_comments(chunk=False, numRows=None) -> pd.DataFrame:
    "Get comments (numRows limits the number of comments returned; chunk processes the csv as chunks)"
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    filepath = folder + 'InputData/PoliticalDiscussion_comments.csv'

    # Load csv in chunks
    if(chunk):
        if(numRows): # If chunk AND numRows 
            comment_chunks = []
            for chunk in pd.read_csv(filepath, chunksize=10000):
                comment_chunks.append(chunk)
            df = pd.concat(comment_chunks, ignore_index=True)
            df = df.sample(numRows, random_state=2)
            return df
        
        comment_chunks = [] # If chunk and NOT numRows
        for chunk in pd.read_csv(filepath, chunksize=10000):
            comment_chunks.append(chunk)
        df = pd.concat(comment_chunks, ignore_index=True)
        return df

    # Load only truncated csv of size numRows
    if(numRows):
        df = pd.read_csv(filepath)
        df = df.sample(numRows, random_state=2)
        return df
    
    return pd.read_csv(filepath) # Return full df if chunk and numRows are both False

def get_submissions(numRows=None) -> pd.DataFrame:
    "Get submissions (numRows limits the number of submissions returned)"
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    filepath = folder + 'InputData/PoliticalDiscussion_submissions.csv'

    # Load only truncated csv of size numRows
    if(numRows): 
        df = pd.read_csv(filepath)
        df = df.sample(numRows, random_state=2)
        return df
    
    return pd.read_csv(filepath) # Return full df

# Uncomment if needed
# def get_cosine_similarity(sentences: list[str]):
#     """Calculate the cosine similarity between two sentences """

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     # model = SentenceTransformer('all-mpnet-base-v2')

#     sentenceEmbeddings = model.encode(sentences)
#     return cosine_similarity([sentenceEmbeddings[0]], [sentenceEmbeddings[1]])[0][0]

# def get_euclidean_distance(sentences: list[str]):
#     """Gets the euclidean distance between two texts"""

#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     sentenceEmbeddings = model.encode(sentences)
#     return np.linalg.norm(sentenceEmbeddings[0] - sentenceEmbeddings[1])

def get_jaccard_similarity(sentences: list[str]):
    """Gets the jaccard similarity between two texts"""

    tokens1 = set(re.findall(r"\b\w+\b", sentences[0].lower()))
    tokens2 = set(re.findall(r"\b\w+\b", sentences[1].lower()))

    if len(tokens1.union(tokens2)) == 0:
        return 0.0

    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))


def get_readability_score(text: str):
    fre_score = textstat.flesch_reading_ease(text)

    return fre_score
    

def add_manual_annotations(inputFile):
    """Add manual annotations to file as a column"""

    df = pd.read_csv(inputFile)

    labels = []

    print('State whether you agree, disagree, or are unsure about the LLM output given the posts you will see.')

    for i, row in enumerate(df['test_posts']):
        print(f'\nRow #{i}..')
        row = ast.literal_eval(row)
        print(f"Test posts: {"\n\n".join(row)}\n")
        print(f"LLM label: {df['LLM_output'][i]}\n")
        label = input("Enter: agree, disagree, or unsure\n")    
        labels.append(label)


    df['true_label'] = labels
    df.to_csv("annotations_with_true_labels.csv", index=False)

def get_annotator_agreement(annotator_df):
    """Calculates the inter-annotator agreement score"""

    agreement_data, categories = aggregate_raters(annotator_df[["annotator_1", "annotator_2", "annotator_3"]])

    # compute Fleiss' kappa
    kappa = fleiss_kappa(agreement_data)

    print("Detected categories:", categories)
    print("Fleiss' κ:", kappa)

    return kappa

def get_majority_vote(df):
    "Get majority vote between 3 annotators, add majority to dataframe column"

    majority_votes = df[["annotator_1", "annotator_2", "annotator_3"]].mode(axis=1)[0]
    df['majority_vote'] = majority_votes

    return df  

def F1_Score_Recall_Precision(true_labels, predicted_labels, debug=True):
    "Compares true labels with the predicted labels"

    precision = precision_score(true_labels, predicted_labels) #, average='weighted')
    recall = recall_score(true_labels, predicted_labels) #, average='weighted')
    f1 = f1_score(true_labels, predicted_labels) #, average='weighted')
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    if debug:
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    return f1, precision, recall, balanced_accuracy

def PreprocessDF(df, columnName: str):
    """Filter dataset to remove NaN content. """

    output_df = (
        df
        .filter(pl.col(columnName).is_not_null())
        .filter(pl.col(columnName) != "NaN")
        .filter(pl.col(columnName) != "nan")
        .filter(pl.col(columnName) != "[deleted]")
        .filter(pl.col(columnName) != "[removed]")
        .filter(pl.col(columnName) != "[deleted by user]")
        .with_columns(
            pl.col(columnName).cast(pl.Utf8),
            pl.col("author").cast(pl.Utf8)
        )
    )

    return output_df


def main():
    annotations_filepath = "RedditDigitalTwin/DigitalTwins_PostPairAnnotations - TrueBias.csv"
    annotations_df = pd.read_csv(annotations_filepath)

    # get_annotator_agreement(annotations_df) # inter-annotator agreement
    df_with_majority_vote = get_majority_vote(annotations_df)
    print(df_with_majority_vote)

if __name__=="__main__":
    main()

