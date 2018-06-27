import nltk
import csv
import numpy as np
import sklearn_crfsuite
import definitions
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.externals import joblib
from sklearn.metrics import  f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict,KFold, cross_val_score, cross_validate, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from	sklearn.base	import	clone
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score, flatten

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
        'klass': tf_idf_clone.predict([word])[0],
        'klass_1': tf_idf_clone_1.predict([word])[0],
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


def clean_labels(old_pred, new_pred, y_test, y_test_new):
    old = []
    new = []
    y = []
    y_new = []

    for string in old_pred:
        temp = []
        for tok in string:
            if tok.find("LOC") != -1 or tok.find("loc") != -1:
                temp.append(1)
            else:
                if tok.find("ORG") != -1 or tok.find('org') != -1 or tok.find('company') != -1:
                    temp.append(2)
                else:
                    if tok.find("PER") != -1 or tok.find("per") != -1 or tok.find("musicartist") != -1:
                        temp.append(3)
                    else:
                        if tok.find("MISC") != -1:
                            temp.append(4)
                        else:
                            temp.append(4)

        old.append(temp)

    for string in new_pred:
        temp = []
        for tok in string:
            if tok.find("LOC") != -1 or tok.find("loc") != -1:
                temp.append(1)
            else:
                if tok.find("ORG") != -1 or tok.find('org') != -1 or tok.find('company') != -1:
                    temp.append(2)
                else:
                    if tok.find("PER") != -1 or tok.find("per") != -1 or tok.find("musicartist") != -1:
                        temp.append(3)
                    else:
                        if tok.find("MISC") != -1:
                            temp.append(4)
                        else:
                            temp.append(4)
        new.append(temp)

    for string in y_test:
        temp = []
        for tok in string:
            if tok.find("LOC") != -1 or tok.find("loc") != -1:
                temp.append(1)
            else:
                if tok.find("ORG") != -1 or tok.find('org') != -1 or tok.find('company') != -1:
                    temp.append(2)
                else:
                    if tok.find("PER") != -1 or tok.find("per") != -1 or tok.find("musicartist") != -1:
                        temp.append(3)
                    else:
                        if tok.find("MISC") != -1:
                            temp.append(4)
                        else:
                            temp.append(4)

        y.append(temp)

    for string in y_test_new:
        temp = []
        for tok in string:
            if tok.find("LOC") != -1 or tok.find("loc") != -1:
                temp.append(1)
            else:
                if tok.find("ORG") != -1 or tok.find('org') != -1 or tok.find('company') != -1:
                    temp.append(2)
                else:
                    if tok.find("PER") != -1 or tok.find("per") != -1 or tok.find("musicartist") != -1:
                        temp.append(3)
                    else:
                        if tok.find("MISC") != -1:
                            temp.append(4)
                        else:
                            temp.append(4)

        y_new.append(temp)

        return  old , new, y, y_new


CONST_WIKI_ALL = "../data/test_data/ritter_ner.tsv"
X_test_final = []
X_test_final_new = []
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
            X_test_final.append(sent2features((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
            # print word[0].split(" ")
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])


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
            X_test_final_new.append(sent2features_new((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            # y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
            # print word[0].split(" ")
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])



# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=20,
#     all_possible_transitions=False,
# )
#
# crf_new = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=20,
#     all_possible_transitions=False,
# )


X_train, X_test, y_train, y_test = train_test_split(X_test_final_new, y_test_final, test_size=0.2, random_state=5)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_test_final_new, y_test_final, test_size=0.2, random_state=5)

# _, _, y_train, y_train_new = clean_labels_in([], [], y_train, y_train_new)

# crf.fit(X_train, y_train)
# crf_new.fit(X_train_new, y_train_new)
# #
# joblib.dump(crf_new, 'crf-suite-new.pkl', compress=9)
# joblib.dump(crf, 'crf-suite-old.pkl', compress=9)
#
ner_new = joblib.load('crf-suite-new.pkl')
ner_old = joblib.load('crf-suite-old.pkl')

new_pred = ner_new.predict(X_test)
old_pred = ner_old.predict(X_test_new)
print(len(y_test), len(y_test_new))
# print (ner_new.score(X_test_new, y_test_new), "new_model")
# print (ner_old.score(X_test, y_test), "old_model")

sorted_labels = definitions.KLASSES.copy()
del sorted_labels[4]

y_new = []
for string in y_test_new:
    temp = []
    for tok in string:
        if tok.find("LOC") != -1 or tok.find("loc") != -1:
            temp.append(1)
        else:
            if tok.find("ORG") != -1 or tok.find('org')!=-1 or tok.find('company') != -1:
                temp.append(2)
            else:
                if tok.find("PER") != -1 or tok.find("per")!= -1 or tok.find("musicartist") != -1:
                    temp.append(3)
                else:
                    if tok.find("MISC") != -1:
                        temp.append(4)
                    else:
                        temp.append(4)

    y_new.append(temp)

new, old, y, h = clean_labels(new_pred,old_pred, y_test, y_test_new)
print(new)
print(old)
print(y)
print(y_new)
print(flat_classification_report(y_new, new, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
print(flat_classification_report(y, old, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
# X_train = np.array(flatten(X_train))
# y_train = np.array(flatten(y_train))
# print(len(X_train), len(y_train))
# X_train_new = np.array(flatten(X_train_new))
# y_train_new = np.array(flatten(y_train_new))

