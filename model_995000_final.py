import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Parameters
embedding_dim = 50
max_length = 100
vocab_size = 10000
test_size = 0.2
filename = "995,000_rows.csv"
chunk_size =10000  # Adjust chunk size based on the system's memory capacity

# Define Label Encoder
label_encoder = preprocessing.LabelEncoder()

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

# Define the model architecture
model = keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load data in chunks
X_accumulated, y_accumulated = [], []


for chunk in pd.read_csv(filename, chunksize=chunk_size):
    # Drop rows with NaN values in the 'content' column
    chunk = chunk.dropna(subset=['content'])

    chunk['type'] = chunk['type'].map({'reliable': 1, 'state': 1, 'political': 1, 'fake': 0, 'satire': 0, 'bias': 0, 'conspiracy': 0, 'junksci': 0, 'hate': 0})

    # Encode labels
    chunk['type'] = label_encoder.fit_transform(chunk['type'])

    
    tokenizer.fit_on_texts(chunk['content'])
    sequences = tokenizer.texts_to_sequences(chunk['content'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    X_accumulated.extend(padded_sequences)
    y_accumulated.extend(chunk['type'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, chunk['type'], test_size=test_size, random_state=42)

    # Train model
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

X_accumulated = np.array(X_accumulated)
y_accumulated = np.array(y_accumulated)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
