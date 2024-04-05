import pandas as pd
from collections import Counter
import re

fake_news_df = pd.read_csv('995,000_rows.csv')

#counting the number of urls
fake_news_df['url_count'] = fake_news_df['content'].str.count('http')
print("Number of URLs in the content:", fake_news_df['url_count'].sum())

#counting the number of dates
fake_news_df['date_count'] = fake_news_df['content'].str.count(r'\d{4}-\d{2}-\d{2}')
print("Number of dates in the content:", fake_news_df['date_count'].sum())

# Function to clean text and tokenize
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text.split()

# Combine content from all rows into a single string
all_text = ' '.join(fake_news_df['content'])

# Clean and tokenize the text
words = clean_text(all_text)

# Count the frequency of each word
word_freq = Counter(words)

# Get the 100 most common words
top_100_words = word_freq.most_common(100)
print("100 most frequent words:", top_100_words)
