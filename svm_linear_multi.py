from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from joblib import dump
from sklearn import svm
# from vocab.Vocab import create_vocab, WordVocab
import pandas as pd
import logging
import re
import json
from preprocess import text_preprocess


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


result = []
logging.basicConfig(level=logging.INFO,
                        filename="./log/10_label_svm.log",
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d]-%(levelname)s: %(message)s')

path = "dataset/"
label = "10_classification"
classification_name = label.replace(" ", "_").replace("/", "_")
model_name = 'models/'+label+'_svm.joblib'
# Data Load

train_path = path + classification_name + '_train_data.tsv'
dev_path = path + classification_name + '_dev_data.tsv'
test_path = path + classification_name + '_test_data.tsv'
data_test = pd.read_csv(test_path, sep='\t')
data_train = pd.read_csv(train_path, sep='\t')
data_dev = pd.read_csv(dev_path, sep='\t')


texts = []
labels = []
for idx in range(data_train.review.shape[0]):

    texts.append(text_preprocess.text_preprocess(data_train.review[idx], method=["stopwords", "lemmatization"]))
    # texts.append(clean_str(data_train.review[idx]))
    labels.append(str(data_train.sentiment[idx]))
    train_split_num = len(data_train)
    dev_split_num = len(data_dev) + train_split_num

vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)

train_vectors = vectorizer.fit_transform(data_train.review)
test_vectors = vectorizer.transform(data_test.review)
dev_vectors = vectorizer.transform(data_dev.review)

classifier_linear = svm.SVC(C=1.0, kernel='linear', random_state=666)
classifier_linear.fit(train_vectors, data_train.sentiment)

p = classifier_linear.predict(test_vectors)
p_dev = classifier_linear.predict(dev_vectors)

test_report_acc = [accuracy_score(data_test.sentiment, p)]
dev_report_acc = [accuracy_score(data_dev.sentiment, p_dev)]

test_report_macro_f1 = [f1_score(data_test.sentiment, p, average='macro')]
dev_report_macro_f1 = [f1_score(data_dev.sentiment, p_dev, average='macro')]

test_report_micro_f1 = [f1_score(data_test.sentiment, p, average='micro')]
dev_report_micro_f1 = [f1_score(data_dev.sentiment, p_dev, average='micro')]

test_classification_report = classification_report(data_test.sentiment, p)
dev_classification_report = classification_report(data_dev.sentiment, p_dev)

logging.info(str(label))
logging.info("Dev Acc : --" + str(dev_report_acc))
logging.info("Test Acc : --" + str(test_report_acc))
print("Dev Acc : --" + str(dev_report_acc))
print("Test Acc : --" + str(test_report_acc))
logging.info("\n-----------------------------\n")

logging.info("Dev Test Macro F1 : --" + str(dev_report_macro_f1))
logging.info("Test Test Macro F1 : --" + str(test_report_macro_f1))
print("Dev Macro F1: --" + str(dev_report_macro_f1))
print("Test Macro F1: --" + str(test_report_macro_f1))
logging.info("\n-----------------------------\n")

logging.info("Dev Test Micro F1 : --" + str(dev_report_micro_f1))
logging.info("Test Test Micro F1 : --" + str(test_report_micro_f1))
print("Dev Micro F1: --" + str(dev_report_micro_f1))
print("Test Micro F1: --" + str(test_report_micro_f1))
logging.info("\n-----------------------------\n")

logging.info("Dev Test classification report : --" + str(dev_classification_report))
logging.info("Test Test classification report : --" + str(test_classification_report))
print("Dev Test classification report --" + str(dev_classification_report))
print("Test Test classification report :" + str(test_classification_report))
logging.info("\n-----------------------------\n")

# save the model
dump(classifier_linear, model_name)

logging.info("\n-----------------------------\n")

logging.info("\n-------------------------------------------END------------------------------------------------------\n")
logging.info("\n-------------------------------------------END------------------------------------------------------\n")
