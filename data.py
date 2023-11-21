import pandas as pd
import numpy as np
import string
import wordninja
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process_dataset(path):
    # load data
    df = pd.read_csv(path)
    df = df[["topic_body", "topics"]].rename(columns={"topic_body": "text"}).dropna()

    # clean text
    exculude = string.punctuation
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].replace('[\n\t\'0-9]', '', regex=True)
    df["text"] = df["text"].apply(lambda s: s.translate(str.maketrans('', '', exculude)))
    df["text"] = df["text"].apply(lambda s: " ".join(wordninja.split(s)))

    # process topics
    df["topics"] = df["topics"].replace('[\',]', '', regex=True)
    for idx, row in df.iterrows():
        topics = df["topics"][idx][1:-1].split()
        df.loc[idx, "topics"] = topics[0]
        if len(topics) > 1:
            df.loc[idx, "multi-topic"] = True
    df["multi-topic"].fillna(False, inplace=True)
    df = df.reset_index(drop=True)
    return df


def tokenize(string, word_level=True, consecutative_tokens=1):
    if word_level:
        tokens = string.split()
        return tokens
    else:
        tokens = [string[i: i + consecutative_tokens] for i in range(0, len(string) - consecutative_tokens)]
        return tokens

def vectorize(tokened_sentances):
    VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']
    unique_tokens = set()


def vectorize_df(df):
    tokens = df["text"].apply(tokenize)


if __name__ == "__main__":
    dataset_path = "Reuters/reuters21578_news.csv"

    ts = time()
    df = process_dataset(dataset_path)
    te = time()
    print("preprocessing took %2.4f sec" % (te - ts))

    vectorize_df(df)
