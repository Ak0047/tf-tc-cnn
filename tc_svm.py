import spacy
from gensim.models import word2vec
import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from gensim.models.word2vec import Text8Corpus
import itertools
import gensim
import nltk
import string
from sklearn.externals import joblib
import input_parser as ip
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords

# nltk.download()

train_data = []
train_labels = []

word2vec_file = "./word2vec_models/default_200/my200.mdl"
# word2vec_file = "./word2vec_models/updated_200/my_data_model.mdl"
input_file = "./data/data.xlsx"


# cachedStopWords = set(stopwords.words("english"))
# cachedStopWords.update(('printer','hp','re','and'))

num_features = 200
# sentences = gensim.models.word2vec.Text8Corpus('text8', max_sentence_length=56)
# model = gensim.models.Word2Vec(sentences, size=num_features, window=5, min_count=5, workers=4)
# model.save("my50.mdl")
# model = gensim.models.Word2Vec.load("my_data_model.mdl")
# w2v = dict(zip(model.index2word, model.syn0))
model = gensim.models.Word2Vec.load(word2vec_file)
# print(w2v)

def vecTransform(X):
    vectors = []
    index2word_set = set(model.index2word)
    for words in X:
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in words.split():
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        vectors.append(featureVec)
    return vectors

# fname = "data.xlsx"
# xl_workbook = xlrd.open_workbook(input_file)
# xl_sheet = xl_workbook.sheet_by_index(0)
#
# for r in range(xl_sheet.nrows):
#     train_data.append(str(xl_sheet.cell_value(r, 0)).lower())
#     train_labels.append(xl_sheet.cell_value(r, 1))
#
# xfm_train_data = train_data[:1200]
# y_train_labels = train_labels[:1200]

input_data = ip.DataParser(input_file)

xfm = vecTransform(input_data.x_text)

classifier = OneVsRestClassifier(LinearSVC(random_state=0))
classifier.fit(np.array(xfm),input_data.y_labels)

# Save Model
joblib.dump(classifier, 'clf-comm.pkl',protocol=2)
clf = joblib.load('clf-comm.pkl')
print("Loaded Model")

# xfm_test_data = train_data[1201:-1]
# y_test_labels = train_labels[1201:-1]
xfm = np.array(vecTransform(input_data.x_text_test))

all_predictions = clf.predict(xfm)

correct_predictions = float(sum(all_predictions == input_data.y_labels_test))
print("Accuracy: {:g}".format(correct_predictions/float(len(input_data.y_labels_test))))

xfm = vecTransform(input_data.x_text)
all_predictions = clf.predict(xfm)

correct_predictions = float(sum(all_predictions == input_data.y_labels))
print("Bias Accuracy: {:g}".format(correct_predictions/float(len(input_data.y_labels))))