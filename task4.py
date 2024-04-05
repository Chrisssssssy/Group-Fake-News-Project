from sklearn.model_selection import train_test_split

# Splitting into training and remaining data (80% / 20%)
train_data, remaining_data = train_test_split(df, test_size=0.2, random_state=42)

# Splitting the remaining data into validation and test sets (50% / 50% of remaining data)
validation_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

# Printing the sizes of each split
print("Training data size:", len(train_data))
print("Validation data size:", len(validation_data))
print("Test data size:", len(test_data))

