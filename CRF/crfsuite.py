import nltk
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.externals import joblib
from sklearn.metrics import  f1_score
from	sklearn.model_selection	import	StratifiedKFold, cross_val_predict,KFold, cross_val_score, cross_validate
from sklearn.preprocessing import MultiLabelBinarizer
from	sklearn.base	import	clone
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import definitions

train_sents = []
train_sents_new = []
test_sents = []
test_sents_new = []
y_train = []
y_test = []

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

with open('../data/test_data/WNUT/17/wnut17train.conll','rb') as tsvin, open('new.csv', 'wb') as csvout:

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
            features = sent2features_new((tree2conlltags(ne_chunk(pos_tag(splitted)))))
            features[0].update({
                # 'klass_1': tf_idf_clone_1.predict([sent])[0],
                # 'klass': tf_idf_clone.predict([sent])[0],
                # 'klass_2': tf_idf_clone_2.predict([sent])[0],
                # 'klass_3': tf_idf_clone_3.predict([sent])[0]
            })
            # print features
            train_sents.append(sent2features((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            train_sents_new.append(features)
            # print "---------"
            # print features
            # print "---------"
            # print(sent)
            # exit(0)
            y_train.append(labels)

            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
                # print word[0].split(" ")
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])
    # except:
    #     print

with open('../data/test_data/WNUT/17/emerging.test.annotated','rb') as tsvin, open('new.csv', 'wb') as csvout:

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
            features = sent2features_new((tree2conlltags(ne_chunk(pos_tag(splitted)))))
            features[0].update({
                # 'klass_1': tf_idf_clone_1.predict([sent])[0],
                # 'klass': tf_idf_clone.predict([sent])[0],
                # 'klass_2': tf_idf_clone_2.predict([sent])[0],
                # 'klass_3': tf_idf_clone_3.predict([sent])[0]
            })

            test_sents.append(sent2features((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            test_sents_new.append(features)
            y_test.append(labels)

            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
                # print word[0].split(" ")
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])
    # except:
    #     print

print "start"
X_train = train_sents
X_train_new = train_sents_new

# print(y_train)
# #
X_test = test_sents
X_test_new = test_sents_new

print "done with 1"

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.088,
    c2=0.002,
    max_iterations=100,
    all_possible_transitions=True,
)

crf_new = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.088,
    c2=0.002,
    max_iterations=100,
    all_possible_transitions=True,
)

crf.fit(X_train, y_train)
print X_train_new
crf_new.fit(X_train_new, y_train)

joblib.dump(crf, 'crf-suite-old.pkl', compress=9)
joblib.dump(crf_new, 'crf-suite-new.pkl', compress=9)

ner_new = joblib.load('crf-suite-new.pkl')
ner_old = joblib.load('crf-suite-old.pkl')

# print (ner_new.score(X_test_new, y_test), "new_model")
# print (ner_old.score(X_test, y_test), "old_model")
#TODO: loop and check in test set for org

new_pred = ner_new.predict(X_test_new)
old_pred = ner_old.predict(X_test)

#TODO: move this into a method

labels = list(ner_new.classes_)
labels.remove('O')
labels.remove('B-creative-work')
labels.remove('I-creative-work')
labels.remove('B-product')
labels.remove('I-product')
labels.remove('B-group')
labels.remove('I-group')

idx = 0
for label in new_pred:
    for tok in label:
        if tok.find('corporation') !=  -1 :
            print label

print "-----------------------------------------"
print(flat_classification_report(y_test, new_pred, labels=labels,digits=3))
print(flat_classification_report(y_test, old_pred, labels=labels, digits=3))
