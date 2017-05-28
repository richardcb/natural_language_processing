import random
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    # init will run first and all other methods will not run unless called
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        # counts how many of the most popular vote were in the list
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# load documents pickle
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

# load word_features pickle
word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    # each word will be included, but no duplicates in the set
    words = word_tokenize(document)
    features = {}
    # for each word in the top 5000 words
    for w in word_features:
        # create boolean of t or f
        features[w] = (w in words)
    return features

# load the featuresets pickle
# load word_features pickle
featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

# shuffle data to lower bias
random.shuffle(featuresets)

# test against the next 1900 words
testing_set = featuresets[10000:]
# train against the first 10000 words
training_set = featuresets[:10000]

# Run all seven classifiers and print their accuracy score:

# load OG naive bayes pickle
open_file = open("pickled_algos/OGnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

# load MultinomialNB pickle
open_file = open("pickled_algos/MultinomialNB_classifier5k.pickle", "rb")
MultinomialNB_classifier = pickle.load(open_file)
open_file.close()

# load BernoulliNB pickle
open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

# load LogisticRegression pickle
open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

# load SGDClassifier pickle
open_file = open("pickled_algos/SGDClassifier_classifier5k.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

# load LinearSVC_classifier pickle
open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

# load LinearSVC_classifier pickle
open_file = open("pickled_algos/NuSVC_classifier5k.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

# selects the classifier with highest accuracy
voted_classifier = VoteClassifier(classifier,
                                  MultinomialNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
