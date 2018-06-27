import csv
import re
#
# with open("/home/hady/nltk_data/corpora/conll2002/eng.train", "r") as fp:
#     reader = csv.reader(fp, delimiter=' ')
#     table = [row for row in reader]
#     with open("/home/hady/nltk_data/corpora/conll2002/eng_new.train", "w+") as fw:
#         # fieldnames = ['class', 'abstract']
#         writer = csv.writer(fw, delimiter=" ")
#         for row in table:
#             if len(row)  > 1:
#                 writer.writerow([row[0], row[1] ,row[-1]])
#             else:
#                 print("hii")
#                 writer.writerow([])

from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk


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

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

X_test_final_new = []
CONST_WIKI_ALL = "../data/test_data/ritter_ner.tsv"
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
            X_test_final_new.append(((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            # y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
            #     print word[0].split(" ")
            if len(word) > 2:
                print word
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])

for x in X_test_final_new:
    print(x)

print(len(X_test_final_new))