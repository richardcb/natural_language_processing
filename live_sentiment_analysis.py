import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
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

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

# PICKLE documents for speed increase
documents = []
for r in short_pos.split('\n'):
    documents.append((r, "pos"))
for r in short_neg.split('\n'):
    documents.append((r, "neg"))

# PICKLE all_words for speed increase
all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())

# convert list all_words to nltk frequency distribution
# convert all words to lowercase
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

"""
Simple ways of printing things:

print the 15 most common words in corpus:
print(all_words.most_common(15))

print the frequency of the words stupid in corpus:
print(all_words["stupid"])
"""
# use up to the top 5000 words to train against
word_features = [w for (w, c) in all_words.most_common(5000)]


def find_features(document):
    # each word will be included, but no duplicates in the set
    words = word_tokenize(document)
    features = {}
    # for each word in the top 5000 words
    for w in word_features:
        # create boolean of t or f
        features[w] = (w in words)
    return features

# finds features by taking the dict of top 3000 words and whether
# or not they are contained in each review. And then the category of
# each of those words. This builds what we can train against,
# or, the existence of words and their impact on whether the review
# is pos/neg
# PICKLE featuresets for speed increase
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# shuffle data to lower bias
random.shuffle(featuresets)

# for positive data:
# train against the first 10000 words
training_set = featuresets[:10000]
# test against the next 1900 words
testing_set = featuresets[10000:]

# # for negative data:
# # train against the 100th and onward word
# training_set = featuresets[100:]
# # test against up to the 100th word
# testing_set = featuresets[:100]

# using naive bayes algorithm:
# posterior = (prior occurences * likelihood) / evidence
# very fast and scalable
classifier = nltk.NaiveBayesClassifier.train(training_set)

"""
Pickle Creation:
# create the pickle
save_classifier = open("naivebayes.pickle","wb")
# define what to include in pickle and where to put inclusions
pickle._dump(classifier, save_classifier)
save_classifier.close()
"""

# # use saved pickle classifier profile rather than a new one
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

# Run all seven classifiers and print their accuracy score:
# PICKLE all classifiers individually for speed increase
print("OG Naive Bayes algorithm accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
print("MNB_classifier algorithm accuracy percent:", (nltk.classify.accuracy(MultinomialNB_classifier, testing_set)) * 100)

# something is wrong with this classifier, gives exactly 50% every run through
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier algorithm accuracy percent:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier algorithm accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier algorithm accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier algorithm accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

# This classifier gives terribad accuracy
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier algorithm accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier algorithm accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier algorithm accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

# selects the classifier with highest accuracy
voted_classifier = VoteClassifier(classifier,
                                  MultinomialNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
print("Voted classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

# testing_set is a dictionary containing t/f examples of pos/neg words.
# passes those testing_set words through find_features
# uses saved pickle profile
# generate output stating classification (pos/neg) & confidence percent
print("Classification:", voted_classifier.classify(testing_set[0][0]), "\nConfidence percent:", voted_classifier.confidence(testing_set[0][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "\nConfidence percent:", voted_classifier.confidence(testing_set[1][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "\nConfidence percent:", voted_classifier.confidence(testing_set[2][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "\nConfidence percent:", voted_classifier.confidence(testing_set[3][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "\nConfidence percent:", voted_classifier.confidence(testing_set[4][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "\nConfidence percent:", voted_classifier.confidence(testing_set[5][0]) * 100)
