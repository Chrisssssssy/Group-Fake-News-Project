import numpy as np
import pandas as pd
import tf_keras as keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


test_data = pd.read_csv("news_sample.csv")  
X_test = test_data['content']  
y_test = test_data['type']


simple_model = keras.models.load_model("simple_model.h5")


advanced_model = keras.models.load_model("advanced_model.h5") #Here tensorflow 2.14 might give an error message

# Make predictions using Simple model
simple_model_predictions = simple_model.predict(X_test)
simple_model_predictions = np.argmax(simple_model_predictions, axis=1)

# Make predictions using Advanced model
advanced_model_predictions = advanced_model.predict(X_test)
advanced_model_predictions = np.argmax(advanced_model_predictions, axis=1)

f1_simple = f1_score(y_test, simple_model_predictions)


f1_advanced = f1_score(y_test, advanced_model_predictions)


print("F1-score for Simple Model:", f1_simple)
print("F1-score for Advanced Model:", f1_advanced)

# Create confusion matrix for Simple model
confusion_matrix_simple = confusion_matrix(y_test, simple_model_predictions)

confusion_matrix_advanced = confusion_matrix(y_test, advanced_model_predictions)


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_simple, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Simple Model')
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_advanced, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Advanced Model')
plt.show()
