import nltk
import sklearn_crfsuite
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.externals import joblib
from sklearn.metrics import  f1_score
from	sklearn.model_selection	import	StratifiedKFold, cross_val_predict,KFold, cross_val_score, cross_validate
from sklearn.preprocessing import MultiLabelBinarizer
from	sklearn.base	import	clone
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

import re
# train_sents = list(nltk.corpus.conll2002.iob_sents('eng.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('eng_new.testb'))
train_sents = []
test_sents = []

with open('/home/hady/nltk_data/corpora/conll2002/eng_new.train','rb') as tsvin, open('new.csv', 'wb') as csvout:
    sent = []
    # try:
    for word in tsvin:
        word = word.split(" ")
        word = [w.replace("\n", "") for w in word]
        word = [w.replace("\r", "") for w in word]


        # splitted = [str.strip(w) for w in word]
        # print word
        # print(splitted)
        if word[0] == '':
            train_sents.append(sent)
            sent = []
        else:
            # if len(word[0].split(" ")) > 1:
                # print word[0].split(" ")
            temp = (word[0], word[1], word[2])
            sent.append(temp)


with open('/home/hady/nltk_data/corpora/conll2002/eng_new.testb','rb') as tsvin, open('new.csv', 'wb') as csvout:
    sent = []
    # try:
    for word in tsvin:
        word = word.split(" ")
        word = [w.replace("\n", "") for w in word]
        word = [w.replace("\r", "") for w in word]


        # splitted = [str.strip(w) for w in word]
        # print word
        # print(splitted)
        if word[0] == '':
            test_sents.append(sent)
            sent = []
        else:
            # if len(word[0].split(" ")) > 1:
                # print word[0].split(" ")
            temp = (word[0], word[1], word[2])
            sent.append(temp)

tf_idf_clone_1 = joblib.load('../one-hot-classifiers/tf-idf+svm_1.pkl')
tf_idf_clone_2 = joblib.load('../one-hot-classifiers/tf-idf+svm_2.pkl')
tf_idf_clone_3 = joblib.load('../one-hot-classifiers/tf-idf+svm_3.pkl')
tf_idf_clone = joblib.load('../multi-class-classifier/tf-idf+svm/tf-idf+svm_new.pkl')

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1].encode("utf-8")
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2].encode("utf-8"),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1].encode("utf-8")
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2].encode("utf-8"),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1].encode("utf-8")
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2].encode("utf-8"),
        })
    else:
        features['EOS'] = True

    return features

def word2features_new(sent, i):
    word = sent[i][0]
    postag = sent[i][1].encode("utf-8")
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2].encode("utf-8"),
        'klass_1': tf_idf_clone_1.predict([word])[0],
        'klass': tf_idf_clone.predict([word])[0],
        'klass_2': tf_idf_clone_2.predict([word])[0],
        'klass_3': tf_idf_clone_3.predict([word])[0],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1].encode("utf-8")
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2].encode("utf-8"),
            '-1:klass': tf_idf_clone.predict([word])[0],
            '-1:klass_1': tf_idf_clone_1.predict([word])[0],
            '-1:klass_2': tf_idf_clone_2.predict([word])[0],
            '-1:klass_3': tf_idf_clone_3.predict([word])[0],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1].encode("utf-8")
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2].encode("utf-8"),
            '+1:klass': tf_idf_clone.predict([word])[0],
            '+1:klass_1': tf_idf_clone_1.predict([word])[0],
            '+1:klass_2': tf_idf_clone_2.predict([word])[0],
            '+1:klass_3': tf_idf_clone_3.predict([word])[0],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2features_new(sent):
    return [word2features_new(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label.encode("utf-8") for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

print "start"
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
# print(y_train)
# #
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
# print "done with 1"
#
X_train_new = [sent2features_new(s) for s in train_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf_new = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
#
crf.fit(X_train, y_train)
crf_new.fit(X_train_new, y_train)

# joblib.dump(crf, 'crf-suite-old.pkl', compress=9)
joblib.dump(crf_new, 'crf-suite-new.pkl', compress=9)

ner_new = joblib.load('crf-suite-new.pkl')
ner_old = joblib.load('crf-suite-old.pkl')

print (ner_new.score(X_test, y_test), "new_model")
print (ner_old.score(X_test, y_test), "old_model")

# print ner_old

# CONST_WIKI_ALL = "/home/hady/PycharmProjects/text-classification-benchmarks/data/test_data/ritter_ner.tsv"
CONST_WIKI_ALL = "../data/test_data/ritter_ner.tsv"

# dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter='\t', skip_header=1)

import csv
X_test_final = []
y_test_final = []


with open(CONST_WIKI_ALL,'rb') as tsvin, open('new.csv', 'wb') as csvout:

    sent = ""
    labels = []
    # try:
    for word in tsvin:
        word = word.split("\t")
        word = [w.replace("\n", "") for w in word]

        if word[0] == '':
            splitted = sent.split(" ")
            splitted = [str.strip(w) for w in splitted]
            # splitted = [re.sub('[^A-Za-z0-9]+', '', w) for w in splitted]
            splitted = [w for w in splitted if len(w) >= 1]
            # print splitted
            X_test_final.append(sent2features_new((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
                # print word[0].split(" ")
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])
    # except:
    #     print

print (ner_new.score(X_test_final, y_test_final), "new_model")

with open(CONST_WIKI_ALL,'rb') as tsvin, open('new.csv', 'wb') as csvout:

    sent = ""
    labels = []
    # try:
    for word in tsvin:
        word = word.split("\t")
        word = [w.replace("\n", "") for w in word]

        if word[0] == '':
            splitted = sent.split(" ")
            splitted = [str.strip(w) for w in splitted]
            # splitted = [re.sub('[^A-Za-z0-9]+', '', w) for w in splitted]
            splitted = [w for w in splitted if len(w) >= 1]
            # print splitted
            X_test_final.append(sent2features((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
                # print word[0].split(" ")
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])

print (ner_old.score(X_test_final, y_test_final), "old_model")
    # y = ner_old.predict(X_test_final)
    # print y
    # print y_test_final

