import pandas as pd
import numpy as np
import string
import wordninja
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def process_dataset(path, processed_path=""):
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
        df.loc[idx, "topics"] = ' '.join(topics)
    df = df.reset_index(drop=True)
    if processed_path:
        df.to_csv(processed_path, index=False)
    return df

def topics_to_vectors(df, min_count_topic):
    topics = df["topics"].apply(str.split, args=(' '))
    mlb = MultiLabelBinarizer(sparse_output=True)
    topics_encoded = pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(topics), index=df.index, columns=mlb.classes_)
    return topics_encoded[topics_encoded.columns[topics_encoded.sum(axis=0) >= 100]]

def tokenize(string, word_level=True, consecutative_tokens=1):
    if word_level:
        tokens = string.split()
        return tokens
    else:
        tokens = [string[i: i + consecutative_tokens] for i in range(0, len(string) - consecutative_tokens)]
        return tokens

def vocabulary(tokened_sentances):
    VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']
    unique_tokens = set()

    for tokens in tokened_sentances:
        unique_tokens.update(tokens)
    VOCAB += sorted(unique_tokens)

    vocab_map = {}
    for i, token in enumerate(VOCAB):
        vocab_map[token] = i

    # vocab_map, unique tokens
    return vocab_map, VOCAB

def vectorize(sentances, vocab_map, word_level, max_length=100):
    n_sequences = len(sentances)
    vectors = np.empty(shape=(n_sequences, max_length), dtype=np.int32)
    vectors.fill(vocab_map["_PAD"])
    for i, sentance in enumerate(sentances):
        tokens = tokenize(sentance, word_level=word_level)
        vectors[i, -len(tokens):] = [vocab_map.get(t, '_UNK') for t in tokens]
    return vectors

def vectorize_df(df):
    tokens = df["text"].apply(tokenize)
    vocab_map, vocab = vocabulary(tokens)
    max_length = tokens.apply(len).max()
    vectors = vectorize(df["text"], vocab_map, True, max_length)
    return vectors, vocab_map


if __name__ == "__main__":
    dataset_path = "Reuters/reuters21578_news.csv"
    processed_path = "text_data.csv"

    # df = process_dataset(dataset_path, processed_path)

    df = pd.read_csv(processed_path)
    # max_words = 400
    # df = df[(df["text"].apply(tokenize).apply(len) <= max_words)]
    # vectors = vectorize_df(df)

    # topics = topics_to_vector(df, 100)
