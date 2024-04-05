import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


original_data = pd.read_csv('news_sample.csv')


extra_data = pd.read_csv("BBC_scraped.csv")

combined_data = pd.concat([original_data, extra_data], ignore_index=True)

# Task 0: Grouping Labelss'
combined_data['type'] = combined_data['type'].map({'reliable': 1, 'state': 1, 'political': 1, 'fake': 0, 'satire': 0, 'bias': 0, 'conspiracy': 0, 'junksci': 0, 'hate': 0})

# Task 1: Feature Extraction
X_combined = combined_data['content'].fillna('')
y_combined = combined_data['type'].fillna('')

# Remove samples with missing values
missing_indices = y_combined[y_combined.isna()].index
X_combined = X_combined.drop(missing_indices)
y_combined = y_combined.drop(missing_indices)

# Check and handle non-numeric values in y_combined
non_numeric_indices = y_combined[~y_combined.apply(lambda x: isinstance(x, (int, float)))].index
X_combined = X_combined.drop(non_numeric_indices)
y_combined = y_combined.drop(non_numeric_indices)

# Making sure y_combined contains only  class labels
y_combined = y_combined.astype(int)

# Splitting data into trainingn and test sets
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Text preprocessing 
vectorizer_combined = HashingVectorizer(n_features=2**10) 
X_train_vec_combined = vectorizer_combined.fit_transform(X_train_combined)
X_test_vec_combined = vectorizer_combined.transform(X_test_combined)

# Train a Logistic Regression model on the combined training data
logreg_model_combined = LogisticRegression()
logreg_model_combined.fit(X_train_vec_combined, y_train_combined)
logreg_pred_combined = logreg_model_combined.predict(X_test_vec_combined)
logreg_accuracy_combined = accuracy_score(y_test_combined, logreg_pred_combined)
print("Logistic Regression Accuracy (with extra reliable data):", logreg_accuracy_combined)





