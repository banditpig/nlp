import pickle
import os.path

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from random import shuffle

# https://www.youtube.com/watch?v=IMKweOTFjXw
# https://www.ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/
positive = []
negative = []
positive_trg = []
negative_trg = []
positive_test = []
negative_test= []

all_trg_data = []
word_features = []

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"

def proc_review(fName, outName):
    outfile = open(outName, "w")

    with open(fName, encoding = "ISO-8859-1") as f:
        for line in f:
            if (line.startswith("<review_text>")):
                done = False
                review = ""
                while (done == False):
                    nextLine = next(f).strip('\n\r')
                    if (nextLine.startswith(("</review_text>"))):
                        done = True
                        outfile.writelines(review.strip('\n\r'))
                        outfile.writelines('\n')
                    else:
                        review = review + nextLine + " "
    outfile.close()


def fill_data():
    with open("posReviews.txt", "r") as f:
        for line in f:
            positive.append((line, POSITIVE))
    with open("negReviews.txt", "r") as f:
        for line in f:
            negative.append((line, NEGATIVE))

    global positive_test
    positive_test = positive[:200]

    global positive_trg
    positive_trg = positive[200:]

    global negative_test
    negative_test = negative[:200]

    global negative_trg
    negative_trg = negative[200:]

    global all_trg_data
    # all_trg_data = positive_trg + negative_trg
    # shuffle(all_trg_data)
    for words, sentiment in positive_trg + negative_trg:
        filtered_words = [w.lower().strip('.').strip(',').strip(';').strip(';').strip(':').strip('"').strip('!').strip('?')
                          for w in words.split() if len(w) >=4]
        all_trg_data.append((filtered_words, sentiment))
    shuffle(all_trg_data)

def get_just_words(data):
    all_words = []
    for(words, sentiment) in data:
        all_words += words
    return all_words;


def extract_features(src_doc):
    document_words = set(src_doc)
    features = {}

    for word in word_features:
        features ['contains(%s)' % word] =  (word in document_words)

    return features



# word_features = nltk.FreqDist( get_just_words(all_trg_data) ).keys()
#
# print (positive_trg[1])
#
# print (extract_features(positive_trg[1]))


def get_classifier(trg_set):

    if (os.path.exists('my_classifier.pickle')):
        clsfr = NaiveBayesClassifier
        f = open('my_classifier.pickle', 'rb')
        clsfr = pickle.load(f)
        f.close()
        return clsfr
    else:
        clsfr = NaiveBayesClassifier(trg_set)
        f = open('my_classifier.pickle', 'wb')
        pickle.dump(clsfr, f)
        f.close()
        return clsfr


# proc_review("negative.txt", "negReviews.txt")
# proc_review("positive.txt", "posReviews.txt")

fill_data()

# for (_, s) in all_trg_data[:100]:
#     print (s)
classifier = get_classifier(all_trg_data[:2800])
print (classifier.accuracy(positive_test))
print (classifier.accuracy(negative_test))


# Accuracy: 0.984

# print("Accuracy: {0}".format(classifier.accuracy(all_trg_data[:250])))
neg = 0
pos = 0
print ("Testing NEGATIVES")
for (review, _) in negative_test:
    if classifier.classify(review) == POSITIVE:
        pos = pos + 1
    else:
        neg = neg + 1

    print( classifier.classify(review) + " pos "+ str(pos) + " neg " + str(neg))

print ("------------------------------------------")
print ("Testing POSITIVES")

neg = 0
pos = 0
for (review, _) in positive_test:
    if classifier.classify(review) == NEGATIVE:
        neg = neg + 1
    else:
        pos = pos + 1

    print(classifier.classify(review) + " pos " + str(pos) + " neg " + str(neg))
# NEGATIVE pos 33 neg 67 1500
# POSITIVE pos 79 neg 21
#
# NEGATIVE pos 39 neg 61  2000
# POSITIVE pos 86 neg 14
classifier.show_informative_features(10)
print (classifier.classify("it was not very good"))
print (classifier.classify("not very good"))
print (classifier.classify("rubbish"))
