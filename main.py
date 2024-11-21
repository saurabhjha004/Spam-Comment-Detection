import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle  # Import pickle to save and load objects

# Load data
data = pd.read_csv("Youtube.csv")
data = data[['CONTENT', 'CLASS']]  # Selecting only CONTENT and CLASS columns

# Mapping class labels
data["CLASS"] = data['CLASS'].map({0: 'NOT A SPAM COMMENT', 1: 'SPAM COMMENT'})

# Prepare data
x = np.array(data['CONTENT'])
y = np.array(data['CLASS'])

# Vectorizing the text data
cv = CountVectorizer()
x = cv.fit_transform(x)

# Splitting data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(xtrain, ytrain)

# Save the trained model and vectorizer to files using pickle
with open('spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)  # Save the model

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)  # Save the vectorizer

# Checking the accuracy of the model on test set
accuracy = model.score(xtest, ytest)
print(f"Model accuracy: {accuracy}")

# Testing the model with user input (for reference)
user_input = input("Enter a comment: ")
input_vector = cv.transform([user_input]).toarray()
prediction = model.predict(input_vector)
print(f"Prediction: {prediction}")
