import pandas as pd
import re
import pandas as pd
import emoji

def remove_urls(text):
    """Removes URLs from text"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_newlines(text):
    """Replaces newlines with a space"""
    return text.replace('\n', ' ').replace('\r', ' ')

def remove_emojis(text):
    """Removes emojis"""
    return emoji.replace_emoji(text, replace='')

def truncate_text(text, word_limit):
    """Truncates text down to the 'word_limit'"""
    words = text.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit])
    return text

def preprocessing_data(df: pd.DataFrame, columnName: str, chunk=False) -> list[str]:
    """"Return a list of preprocessed posts, removes nan and deleted text, URLs, emojis and truncates text (if needed).
    'columnName' should be either 'body' (comments file) or 'selftext' (submissions file) """

    # Drop rows with no text
    df = df.dropna(subset=[columnName])

    # Convert dataframe column to string
    df[columnName] = df[columnName].astype(str)
    df = df[(df[columnName] != 'nan') & (df[columnName] != '[deleted]') & (df[columnName] != '[removed]') & (df[columnName] != '[deleted by user]')]

    return df






