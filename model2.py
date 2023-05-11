import numpy as np
import pandas as pd

# For text Pre-processing 
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string

# For import the pkl file
import pickle

# implementation
ps = PorterStemmer()

data = pd.read_csv('fourdata-different.csv')
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data.head()

len(data.category.value_counts())

data.category.unique()

data.isnull().sum()

data.shape

data.drop_duplicates(inplace=True)
data.shape

print(data['description'].apply(lambda x: len(x.split(' '))).sum())

#Text processing

def clean_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
data['description'] = data['description'].apply(clean_text)

print(data['description'].apply(lambda x: len(x.split(' '))).sum())


# Train & test Split and Labelling

X = data.description
y = data.category
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
v = dict(zip(list(y), data['category'].to_list()))

# Applying Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

lr = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression()),
              ])
allset = lr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)

# print(f"Accuracy is :{accuracy_score(y_pred1,y_test)}")
percentage = accuracy_score(y_pred1,y_test)
if percentage > 0.8:
    print(percentage)
else :
    print("others")

# RESULT CHECKING Accuraccy = 83%

# Testing

var = ['javascript is know language for me']

res = lr.predict(var)

# New alteration for checking the accuraccy
check = accuracy_score(res,var)
if check >0.8 :
 print(v[res[0]])
else :
 print("Others")

print(check)
print(res)

# Import the pkl
with open('model22.pkl','wb') as f:
    pickle.dump(lr, f)

