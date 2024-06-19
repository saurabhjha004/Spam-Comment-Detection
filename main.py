import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


data = pd.read_csv("Youtube.csv") 

# Since we only require content and class columns, we will update our existing datafram to the following
data = data[['CONTENT','CLASS']]

# We will map 0 to not spam and 1 to span in class column
data["CLASS"] = data['CLASS'].map({0:'NOT A SPAM COMMENT', 1: 'SPAM COMMENT'})

x = np.array(data['CONTENT'])
y = np.array(data['CLASS'])

"""
As the output of this problem will either be 0 or 1,i.e, the problem of binary classification,
we can use the Bernoulli Naive Bayes algorithm to train the model:
"""

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.2,random_state= 42)

model = BernoulliNB()
model.fit(xtrain,ytrain)

print(model.score(xtest,ytest))

# We will check the model by giving spam and non-spam comments

S = input("Enter a comment : ")
d = cv.transform([S]).toarray()
print(model.predict(d))
