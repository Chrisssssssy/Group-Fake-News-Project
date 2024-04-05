import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


url = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"

df = pd.read_csv(url)

def clean_text(text): 
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    text = re.sub(r"\b(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})\b", "<DATE>", text)
    text = re.sub(r'\b(?:https?://|www\.)\S+', "<URL>", text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def stem_text(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

df['cleaned_text'] = df['content'].apply(clean_text)
df['tokenized_text'] = df['content'].apply(tokenize_text)
df['stemmed_text'] = df['tokenized_text'].apply(stem_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_text = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Compute the size of vocabulary before and after removing stopwords
vocabulary_before = len(set([word for sublist in df['tokenized_text'] for word in sublist]))
vocabulary_after = len(set([word for sublist in filtered_text for word in sublist]))
reduction_rate_stopwords = (vocabulary_before - vocabulary_after) / vocabulary_before

# Compute the size of vocabulary before and after stemming
vocabulary_after_stemming = len(set([word for sublist in df['stemmed_text']for word in sublist]))
reduction_rate_stemming = (vocabulary_after - vocabulary_after_stemming) / vocabulary_after


#print(f"Cleaned Text:\n{df['cleaned_text']}\n")
#print(f"Tokenized Text:\n{df['tokenized_text']}")
print(f"Stemmed Text:\n{df['stemmed_text']}")


print("Vocabulary size before removing stopwords:", vocabulary_before)
print("Vocabulary size after removing stopwords:", vocabulary_after)
print("Reduction rate after removing stopwords:", reduction_rate_stopwords)
print("Vocabulary size after stemming:", vocabulary_after_stemming)
print("Reduction rate after stemming:", reduction_rate_stemming)


from sklearn.model_selection import train_test_split

# Splitting into training and remaining data (80% / 20%)
train_data, remaining_data = train_test_split(df, test_size=0.2, random_state=42)

# Splitting the remaining data into validation and test sets (50% / 50% of remaining data)
validation_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

# Printing the sizes of each split
print("Training data size:", len(train_data))
print("Validation data size:", len(validation_data))
print("Test data size:", len(test_data))
