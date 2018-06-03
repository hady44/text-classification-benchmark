import nltk
import sklearn_crfsuite
import eli5
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.externals import joblib
from nltk.chunk import conlltags2tree, tree2conlltags
import re
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
# print(train_sents)
tf_idf_clone = joblib.load('tf-idf+svm.pkl')
def word2features(sent, i):
    word = sent[i][0]
    print tf_idf_clone.predict([word])[0]
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
            '-1:klass': tf_idf_clone.predict([word1])[0],
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
            '+1:klass': tf_idf_clone.predict([word1])[0],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label.encode("utf-8") for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# X_train = [sent2features(s) for s in train_sents]
# y_train = [sent2labels(s) for s in train_sents]
#
# X_test = [sent2features(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]


# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=20,
#     all_possible_transitions=False,
# )
# crf.fit(X_train, y_train);


# joblib.dump(crf, 'crf-suite-new.pkl', compress=9)
ner_old = joblib.load('crf-suite-old.pkl')
ner_new = joblib.load('crf-suite-new.pkl')

# print ner_old

CONST_WIKI_ALL = "/home/hady/PycharmProjects/text-classification-benchmarks/data/ritter_ner.tsv"

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
            X_test_final.append(sent2features((tree2conlltags(ne_chunk(pos_tag(splitted))))))
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


# with open(CONST_WIKI_ALL,'rb') as tsvin, open('new.csv', 'wb') as csvout:
#     tsvin = csv.reader(tsvin, delimiter='\t')
#     for word in tsvin:
#         print word
#         break
#     csvout = csv.writer(csvout)
#     idx = 0
#     sent = ""
#     labels = []
#     for row in tsvin:
#         # X_test_final[idx].append(row[0])
#         # y_test_final[idx].append(row[1]
#         if len(str.strip(row[0])) < 1:
#             words = nltk.word_tokenize(sent)
#             words = [word.lower() for word in words if word.isalpha()]
#
#             X_test_final.append(sent2features((tree2conlltags(ne_chunk(pos_tag(word))))))
#             y_test_final.append(labels)
#             # print sent
#             # print "------------------------------"
#             sent = ""
#             labels = []
#         else:
#             row[0] = row[0].replace('"', '')
#             words = (re.findall(r"\w+", row[0]))
#             if len(words) > 10:
#                 print row[0]
#                 break
#             sent = sent +" "+ str.strip(row[0])
#             labels.append(row[1])

# print(pos_tag(word_tokenize("hady")))
# print train_sents[0]
# print y_train[0]
# print(len(X_test_final[1]), len(y_test_final[1]))
for idx in range(len(X_test_final)):
    x = X_test_final[idx]
    y = y_test_final[idx]
    if len(x) != len(y):
        print "err"
# print ner_old.predict(sent2features("Hady is a good boy"))
print ner_old.score(X_test_final, y_test_final), "old_model"
print ner_new.score(X_test_final, y_test_final), "new_model"

# print crf.score(X_test_final, y_test_final), "new_model" #new crf

# print(X_train[0])

# print(y_train[0])


