from flask import Flask, render_template,redirect,request
ml = Flask(__name__)

# Machine Learning implememtation

import string
import pickle
import numpy as np
import pandas as pd

# for count occurrence
from collections import Counter

# Natural language imports

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
# Function for the Text
def transform_text(text):
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

# Loading the datamodel
with open('model22.pkl','rb') as f:
    obj = pickle.load(f)

@ml.route('/')
def home():
    return render_template('index.html')

@ml.route('/predict',methods=['POST'])
def des():
    if request.method == "POST":
        description = request.form["des"]
        category = transform_text(description)
        def Convert(string):
            li = list(string.split(" "))
            return li
  
        ls = []
        ls = Convert(category)

        result =obj.predict(ls)
        print(result)
        
        # Function for more occurrence
        def most_frequent(List):
            occurence_count = Counter(List)
            return occurence_count.most_common(1)[0][0]

        
        occur_word = most_frequent(result)

        # occurence_number = np.count_nonzero(occur_word > 2)
        occurence_number = result.tolist().count(occur_word)
        print(occur_word)
        print(occurence_number)

        if occurence_number >= 2:
            
            return occur_word
        else :
            return 'others'

    return 'allDone'

ml.run(debug=True)