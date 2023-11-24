#!/usr/bin/env python
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.regularizers as regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data import vectorize_df, tokenize, vocabulary
from data import topics_to_vectors

df = pd.read_csv("text_data.csv")
df.head()

# process text


max_words = 400

df = df[df["text"].apply(str.split).apply(len) <= max_words]

vectors, vocab_map = vectorize_df(df)
print(vectors.shape)

# process topics
min_topic_count = 100

topics = topics_to_vectors(df, min_topic_count).values
print(topics.shape)

# remove lines with no topics
mask = topics.sum(axis=1) != 0
topics = topics[mask]
vectors = vectors[mask]
df = df[mask]


# Model

VOCAB_SIZE = len(vocab_map)
INPUT_SIZE = max_words

model_simple = models.Sequential([
    layers.Dense(256, input_shape=(vectors.shape[1], )),
    layers.Dropout(0.2),
    layers.Dense(topics.shape[1], activation="softmax")
])

model_gru = models.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=256, input_length=INPUT_SIZE),
    layers.Dropout(0.2),
    layers.GRU(256, return_sequences=True),
    layers.GRU(64, return_sequences=True),
    layers.Flatten(),
    layers.Dense(topics.shape[1], activation="softmax")
])

model_gru_small = models.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=INPUT_SIZE),
    layers.Dropout(0.5),
    layers.GRU(64, return_sequences=True),
    layers.GRU(32, return_sequences=True),
    layers.Flatten(),
    layers.Dense(topics.shape[1], activation="softmax")
])

model_dense = models.Sequential([
    layers.Dense(256, kernel_regularizer=regularizers.L1L2(), activation='relu', input_shape=(vectors.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(256, kernel_regularizer=regularizers.L1L2(), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, kernel_regularizer=regularizers.L1L2(), activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(topics.shape[1], activation="softmax")
])

model_dense_embed = models.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=512, input_length=INPUT_SIZE),
    layers.Flatten(),
    layers.Dense(1024, kernel_regularizer=regularizers.L1L2(), activation='relu'),
    layers.Dense(256, kernel_regularizer=regularizers.L1L2(), activation='relu'),
    layers.Dense(topics.shape[1], activation="softmax")
])


model = model_dense_embed

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


x_train, x_test, y_train, y_test = train_test_split(vectors, topics, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model.evaluate(x_test, y_test)


history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))


def plot_learning_curves(history, title=""):
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle(title, fontsize="x-large")

    ax1.plot(epochs, acc, label="Entraînement")
    ax1.plot(epochs, val_acc, label="Validation")
    ax1.set_title("Accuracy - Données entraînement vs. validation.")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, loss, label="Entraînement")
    ax2.plot(epochs, val_loss, label="Validation")
    ax2.set_title("Perte - Données entraînement vs. validation.")
    ax2.set_ylabel('Perte')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    if title:
        plt.savefig(title)
    fig.show()


plot_learning_curves(history, "hist.png")
