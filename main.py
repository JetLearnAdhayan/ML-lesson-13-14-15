#NLP - natural language processing

import pandas as pd
from sklearn.model_selection import train_test_split 

data = pd.read_csv("train.txt", sep=";", names = ["text","label"],nrows=5000)

data["label"] = data["label"].replace({"sadness": 1,"anger": 1, "fear": 1 , "joy": 0, "love":0,"surprise":0})

X = data["text"]
Y = data["label"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re #regular expressions

wne = WordNetLemmatizer()

def transform(data):
    corpus = []
    for i in data:
        newi = re.sub("^a-zA-Z"," ", i)
        newi = newi.lower()
        newi = newi.split()

        list1 = [wne.lemmatize(word) for word in newi if word not in stopwords.words("english")]
        corpus.append(" ".join(list1))

    return corpus

X_train_corpus = transform(X_train)
X_test_corpus = transform(X_test)

print(X_train_corpus[0:5])
