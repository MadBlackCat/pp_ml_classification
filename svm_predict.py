from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import dump, load
from sklearn import svm

import pandas as pd
import numpy as np
import logging
import time
import re
import os
import pickle


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    try:
        string = re.sub(r'\\', "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()
    except:
        print(string)


label_list = ["Introductory", "First Party", "Third Party", "Cookies", "Data Retention", "User Right",
              "Data Transfer", "Internal and Specific Audiences", "Policy Change", "Privacy Contact Information"]
result = []
path = "validation_data/"
label = "10_classification"
classifier_linear = load('models/10_label_svm.joblib')
validation_data = os.listdir(path)

vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)
label = "10_classification"
classification_name = label.replace(" ", "_").replace("/", "_")
train_path = "10-classification/" + classification_name + '_train_data.tsv'
data_train = pd.read_csv(train_path, sep='\t')
train_vectors = vectorizer.fit_transform(data_train.review)
# Data Load
for pp in validation_data:
    test_path = path + pp
    with open(test_path, "r", encoding="utf8") as f:
        data = f.read()
    data = [i.strip() for i in data.split('\n')
            if len(''.join(list(filter(str.isalpha, i)))) > 1]
    data_test = pd.DataFrame(data, columns=['review'])
    test_vectors = vectorizer.transform(data_test.review)
    p = classifier_linear.predict(test_vectors)
    result = [r"<html><body>"]
    for i, j in enumerate(p):
        result.append(r"<h4>"+label_list[j]+r"</h4>")
        result.append(r"<p>"+data[i]+r"</p>")
    result.append(r"</body></html>")
    with open(test_path.replace(".txt", ""), "w", encoding="utf-8") as f:
        f.write("\n".join(result))


