import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Parameters
embedding_dim = 50
max_length = 100
vocab_size = 10000
test_size = 0.2
filename = "news_sample.csv"
chunk_size = 10000  # Adjust chunk size based on your system's memory capacity

# Define Label Encoder
label_encoder = preprocessing.LabelEncoder()

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

# Define model architecture
model = keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load data in chunks
X_accumulated, y_accumulated = [], []

for chunk in pd.read_csv(filename, chunksize=chunk_size):
    # Drop rows with NaN values in the 'content' column
    chunk = chunk.dropna(subset=['content'])

    chunk['type'] = chunk['type'].map({'reliable': 1, 'state': 1, 'political': 1, 'fake': 0, 'satire': 0, 'bias': 0, 'conspiracy': 0, 'junksci': 0, 'hate': 0})

    # Encode labels
    chunk['type'] = label_encoder.fit_transform(chunk['type'])

    # Tokenization
    sequences = tokenizer.texts_to_sequences(chunk['content'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    X_accumulated.extend(padded_sequences)
    y_accumulated.extend(chunk['type'])

X_accumulated = np.array(X_accumulated)
y_accumulated = np.array(y_accumulated)

X_train, X_test, y_train, y_test = train_test_split(X_accumulated, y_accumulated, test_size=test_size, random_state=42)

model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Load the LIAR dataset
liar_data = pd.read_csv("liar_dataset/test.tsv", sep='\t', header=None)

liar_data[1] = liar_data[1].map({'true': 1, 'mostly-true': 1, 'half-true': 1, 'barely-true': 0, 'false': 0, 'pants-fire': 0})

# Task 1: Feature Extraction
X_liar = liar_data[2].fillna('')
y_liar = liar_data[1].fillna('')

# Tokenization and padding
sequences_liar = tokenizer.texts_to_sequences(X_liar)
X_liar_padded = pad_sequences(sequences_liar, maxlen=max_length, padding='post', truncating='post')

# Predict probabilities
liar_pred_prob = model.predict(X_liar_padded)

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
liar_pred = (liar_pred_prob > 0.5).astype(int)

# Evaluate accuracy
liar_accuracy = accuracy_score(y_liar, liar_pred)
print("Model Accuracy on LIAR dataset:", liar_accuracy)
