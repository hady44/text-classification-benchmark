import nltk
import sklearn_crfsuite
from numpy.ma import average
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.externals import joblib
from sklearn.metrics import  f1_score
from	sklearn.model_selection	import	StratifiedKFold, cross_val_predict,KFold, cross_val_score, cross_validate, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from	sklearn.base	import	clone
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import  CRF.definitions as definitions


lancaster_stemmer = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case=True, strip_handles=False, reduce_len=False)
stop = set(stopwords.words('english'))

def get_tuples(dspath):
    sentences = []
    s = ''
    tokens = []
    ners = []
    poss = []
    tot_sentences = 0
    ners_by_position = []
    index = 0
    with open(dspath) as f:
        for line in f:
            if line.strip() != '':
                token = line.split('\t')[0].decode('utf-8')
                ner = line.split('\t')[1].replace('\r', '').replace('\n', '').decode('utf-8')
                '''
                if ner in definitions.NER_TAGS_ORG:
                    ner = 'ORG'
                elif ner in definitions.NER_TAGS_LOC:
                    ner = 'LOC'
                elif ner in definitions.NER_TAGS_PER:
                    ner = 'PER'
                else :
                    ner = 'O'
                '''
                #ners_by_position.append([index, len(token), ner])
                index += len(token) + 1
            if line.strip() == '':
                if len(tokens) != 0:
                    #poss = [x[1].decode('utf-8') for x in nltk.pos_tag(nltk.word_tokenize(s[:-1]))]
                    poss = [x[1].decode('utf-8') for x in nltk.pos_tag(tknzr.tokenize(s[:-1]))]


                    #if len(poss) == len(tokens): # tokenization doesnt affect position of NERs, i.e., same tokenization
                    sentences.append(zip(tokens, poss, ners))
                    #else:
                    #    aux = 0
                    #    for i in range(len()):
                    #        if aux <= tokens[i]

                    tokens = []
                    ners = []
                    s = ''
                    tot_sentences += 1
            else:
                s += token + ' '
                tokens.append(token)
                ners.append(ner)

    return sentences


dataset_ritters_train = get_tuples('../../data/test_data/ritter_ner.tsv')

train_sents = dataset_ritters_train

tf_idf_clone_1 = joblib.load('../../one-hot-classifiers/tf-idf+svm_1.pkl')
tf_idf_clone_2 = joblib.load('../../one-hot-classifiers/tf-idf+svm_2.pkl')
tf_idf_clone_3 = joblib.load('../../one-hot-classifiers/tf-idf+svm_3.pkl')
tf_idf_clone = joblib.load('../../multi-class-classifier/tf-idf+svm/tf-idf+svm_new.pkl')


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
        'stop_word': word in stop,
        'hyphen': '-' in word,
        'size_small': True if len(word) <= 2 else False,
        'stemmer_lanc': lancaster_stemmer.stem(word),
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
        'stop_word': word in stop,
        'hyphen': '-' in word,
        'size_small': True if len(word) <= 2 else False,
        'stemmer_lanc': lancaster_stemmer.stem(word),
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


def f1_score_mod(y_true, y_pred):

    # TODO: move this into a method

    new = []
    y = []
    cnt = 0
    for string in y_pred:
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

    for string in y_true:
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

    return flat_f1_score(y, new, average='weighted', labels=sorted_labels)


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2features_new(sent):
    return [word2features_new(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label.encode("utf-8") for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X_train = [sent2features(s) for s in train_sents]
X_train_new = [sent2features_new(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
# X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train_new, y_train, test_size=0.2, random_state=5)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=5)
# X_test = [sent2features(s) for s in test_sents]
# X_test_new = [sent2features_new(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]

print "start"

# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.088,
#     c2=0.002,
#     max_iterations=100,
#     all_possible_transitions=True,
# )
#
# crf_new = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.088,
#     c2=0.002,
#     max_iterations=100,
#     all_possible_transitions=True,
# )
#
# crf.fit(X_train, y_train)
# crf_new.fit(X_train_new, y_train)
#
# joblib.dump(crf, 'crf-suite-old.pkl', compress=9)
# joblib.dump(crf_new, 'crf-suite-new.pkl', compress=9)

ner_new = joblib.load('crf-suite-new.pkl')
ner_old = joblib.load('crf-suite-old.pkl')

labels = list(ner_old.classes_)
labels.remove('O')
labels.remove('B-facility')
# labels.remove('I-company')
labels.remove('I-facility')
labels.remove('B-movie')
labels.remove('I-movie')
labels.remove('B-musicartist')
labels.remove('I-musicartist')
labels.remove('B-other')
labels.remove('I-other')
labels.remove('B-product')
labels.remove('I-product')
labels.remove('B-sportsteam')
labels.remove('I-sportsteam')
labels.remove('B-tvshow')
if 'I-tvshow' in labels:
    labels.remove('I-tvshow')

f1_scorer = make_scorer(f1_score_mod)
f1_scorer_mod = make_scorer(flat_f1_score,
                        average='weighted', labels=labels)

# y_pred_new = ner_new.predict(y_test_new)
# y_pred_old = ner_old.predict(y_test)
# print(y_pred_old)
# print(y_test)
# print(flat_classification_report(y_test_new, y_pred_new, labels=labels,digits=3))
# print(flat_classification_report(y_test, y_pred_old, labels=labels, digits=3))

sorted_labels = definitions.KLASSES.copy()
del sorted_labels[4]

scores_new = cross_val_score(ner_new, X_train_new, y_train, scoring=f1_scorer, cv=5)
scores_old = cross_val_score(ner_old, X_train, y_train, scoring=f1_scorer, cv=5)

print(scores_new)
print "----------------------"
print(scores_old)

print "----------------------"

scores_new = cross_val_score(ner_new, X_train_new, y_train, scoring=f1_scorer_mod, cv=5)
scores_old = cross_val_score(ner_old, X_train, y_train, scoring=f1_scorer_mod, cv=5)

print(scores_new)
print "----------------------"
print(scores_old)
print "----------------------"
print(labels)
