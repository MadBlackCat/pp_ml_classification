from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import svm
from sklearn.metrics import classification_report
from vocab.Vocab import create_vocab, WordVocab
import pandas as pd
import numpy as np


from bs4 import BeautifulSoup
import logging
import time
import re
import pickle

# Create feature vectors
# vectorizer = TfidfVectorizer(min_df = 5,
#                              max_df = 0.8,
#                              sublinear_tf = True,
#                              use_idf = True)
# a = trainData['Content']
# b = testData['Content']
# train_vectors = vectorizer.fit_transform(trainData['Content'])
# test_vectors = vectorizer.transform(testData['Content'])


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

#
# label_list = ["Introductory/Generic", "First Party Collection", "First Party Use", "Legal Basis", "Cookies and Similar Technologies",
#              "Third Party Sharing and Collection", "Links", "User Control",  "Data Security",  "Data Retention",
#              "Data Transfer",  "Internal and Specific Audiences",   "User Right", "Policy Change",  "Privacy Contact Information"]

label_list = ["Introductory", "First Party", "Third Party", "Cookies", "Data Retention", "User Right",
"Data Transfer", "Internal and Specific Audiences", "Policy Change", "Privacy Contact Information"]
result = []
logging.basicConfig(level=logging.INFO,
                        filename="./svm.log",
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d]-%(levelname)s: %(message)s'
                        )

# label = "15_classification"
# classification_name = label.replace(" ", "_").replace("/", "_")
# test_path = './data/' + classification_name + '_test_data.tsv'
# data_test = pd.read_csv(test_path, sep='\t')
path = "un_downsampling/"
# label_list = [str(i) for i in range(0, 10)]
for label, label_name in enumerate(label_list):
    # label = str(label)
    label = label_name
    classification_name = label.replace(" ", "_").replace("/", "_")
    train_path = path + classification_name + '_train_data.tsv'
    dev_path = path + classification_name + '_dev_data.tsv'
    test_path = path + classification_name + '_test_data.tsv'
    data_test = pd.read_csv(test_path, sep='\t')
    data_train = pd.read_csv(train_path, sep='\t')
    data_dev = pd.read_csv(dev_path, sep='\t')

    # data_train = pd.DataFrame(pd.concat([data_train, data_dev], ignore_index=True))
    texts = []
    labels = []
    for idx in range(data_train.review.shape[0]):
        # text = BeautifulSoup(data_train.review[idx]).decode('utf-8')
        texts.append(clean_str(data_train.review[idx]))
        labels.append(str(data_train.sentiment[idx]))
        train_split_num = len(data_train)
        dev_split_num = len(data_dev) + train_split_num
    # x_train = pd.DataFrame(texts[:dev_split_num])
    # y_train = pd.DataFrame(labels[:dev_split_num])
    # x_test = pd.DataFrame(texts[dev_split_num:])
    # y_test = pd.DataFrame(labels[dev_split_num:])
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)

    train_vectors = vectorizer.fit_transform(data_train.review)
    test_vectors = vectorizer.transform(data_test.review)
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, data_train.sentiment)
    # prediction_linear = np.max(classifier_linear.predict_log_proba(test_vectors), axis=1)
    p = classifier_linear.predict(test_vectors)
    # logging.info("\n--------------------"+label+"-------------------------")
    # report = classification_report(data_test.sentiment, prediction_linear)
    report2 = [precision_score(data_test.sentiment, p),
               recall_score(data_test.sentiment, p),
               f1_score(data_test.sentiment, p)]
    # result.append(list(prediction_linear))
    logging.info(str(label_name))
    logging.info(report2)
    logging.info("\n-----------------------------\n")
    # logging.info(str(report2))
# result = np.argmax(np.transpose(np.array(result)), axis=1)

# print()
# print(accuracy_score(data_test.sentiment, result))
logging.info("\n-----------------------------------------------------END------------------------------------------------------\n")
logging.info("\n-----------------------------------------------------END------------------------------------------------------\n")
