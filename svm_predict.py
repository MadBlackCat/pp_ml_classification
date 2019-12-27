from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import dump, load
from sklearn import svm
from preprocess import merge
from preprocess import predict_precess
import pandas as pd
import numpy as np
import logging
import time
import re
import os
import pickle
import json
import string


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


def takeOne(ele):
    return ele[0]


punc = string.punctuation
result = []
path = "validation_txt/"
label = "10_classification"
classifier_linear = load('models/' + label + '_svm.joblib')

label_list = ["Introductory", "How We Collect and Use Your Information", "Cookies and Similar Technologies",
              "Third Party Sharing and Collection", "What You Can Do", "How We Protect Your Information",
              "Data Retention", "International Data Transfer", "Specific Audiences", "Policy Change",
              "Contact Information"]

validation_data = [i for i in os.listdir(path) if i[-4:] == ".txt"]

vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)

label = "10_classification"
classification_name = label.replace(" ", "_").replace("/", "_")
train_path = "dataset/" + classification_name + '_train_data.tsv'
data_train = pd.read_csv(train_path, sep='\t')
train_vectors = vectorizer.fit_transform(data_train.review)
# Data Load
for pp in validation_data:
    # pp = "com.irexdroid.rapidmemory.html.txt"
    test_path = path + pp
    with open(test_path, "r", encoding="utf8") as f:
        data = f.read()

    data_origin, data = predict_precess.predict_preprocess(data)
#######################################################################################
    data_test = pd.DataFrame(data, columns=['review'])
    test_vectors = vectorizer.transform(data_test.review)
    p = classifier_linear.predict(test_vectors)
######################################################################################
    result_dict = {data[i]: value for i, value in enumerate(p)}
    result = []
    for i, label in enumerate(label_list):
        if i in p:
            result.append([key for key, value in result_dict.items() if value == i])
        else:
            result.append(None)
##############################################################################################################
    result = []
    for i, label in enumerate(label_list):
        if i in p:
            result.append([data_origin[key] for key, value in enumerate(p) if value == i])
        else:
            result.append(None)

    html_result = [r"<html><body><div style ='"
                   r"padding: 100px;"
                   r"width: 60%;"
                   r"margin: auto; "
                   r"position: absolute;"
                   r"top: 0;"
                   r"left: 0;"
                   r"right: 0;"
                   r"bottom: 0;"
                   r"font-size: 120%;'>"
                   r"<h1 style='font-size: 200%;"
                   r"text-align:center;'>Privacy Policy</h1>"
                   ]
    for i, para_list in enumerate(result):

        if para_list is not None:
            html_result.append(r"<h2> " + label_list[i] + r"</h2>")
            for para in para_list:
                if len(para) > 1:
                    if para[0][-1:] == ':':
                        html_result.append(r"<p>" + para[0] + r"</p>")
                        html_result.append(r"<ul>")
                        for j in para[1:]:
                            # 新添加约束条件，有可能会去掉大量有关信息

                            html_result.append(r"<li>" + j + r"</li>")
                        html_result.append(r"</ul>")
                    else:
                        html_result.append(r"<ul>")
                        for j in para[1:]:
                            # 新添加约束条件，有可能会去掉大量有关信息
                            if not predict_precess.unrelated (j) and len (j.strip ()) > 0:
                                html_result.append(r"<li>" + j + r"</li>")
                        html_result.append(r"</ul>")
                else:
                    for j in para:
                        # if i == 10:
                        #     html_result.append(r"<p>" + j + r"</p>")
                        if not predict_precess.unrelated(j) and len(j.strip()) > 0:
                            html_result.append(r"<p>" + j + r"</p>")

    html_result.append(r"</div></body></html>")
    with open("./validation_data/" + pp.replace(".txt", ""), "w", encoding="utf-8") as f:
        f.write("\n".join(html_result))


