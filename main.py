#NLP - natural language processing

import pandas as pd
from sklearn.model_selection import train_test_split 

data = pd.read_csv("train.txt", sep=";", names = ["text","label"],nrows=5000)

data["label"] = data["label"].replace({"sadness": 1,"anger": 1, "fear": 1 , "joy": 0, "love":0,"surprise":0})

X = data["text"]
Y = data["label"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

