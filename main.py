import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("Youtube.csv")
data = data[['CONTENT', 'CLASS']] # Selecting only CONTENT and CLASS columns after loading data

# Mapping class labels
data["CLASS"] = data['CLASS'].map({0: 'NOT A SPAM COMMENT', 1: 'SPAM COMMENT'})

x = np.array(data['CONTENT'])
y = np.array(data['CLASS'])

cv = CountVectorizer()
x = cv.fit_transform(x)
# Splitting data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Bernoulli Naive Bayes model
model = BernoulliNB()

# Training the model
model.fit(xtrain, ytrain)

# Checking the accuracy of the model on test set
accuracy = model.score(xtest, ytest)
print(f"Model accuracy: {accuracy}")

# Testing the model with user input
user_input = input("Enter a comment: ")
input_vector = cv.transform([user_input]).toarray()
prediction = model.predict(input_vector)
print(f"Prediction: {prediction}")
