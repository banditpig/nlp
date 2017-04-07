# import nltk
# from nltk.classify import NaiveBayesClassifier
from textblob.classifiers import NaiveBayesClassifier
# from nltk.tokenize import sent_tokenize, word_tokenize

# https://www.youtube.com/watch?v=IMKweOTFjXw
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
    positive_test = positive[:100]

    global positive_trg
    positive_trg = positive[100:]

    global negative_test
    negative_test = negative[:100]

    global negative_trg
    negative_trg = negative[100:]

    for words, sentiment in positive_trg + negative_trg:
        filtered_words = [w.lower().strip('.').strip(',').strip(';').strip(';').strip(':').strip('"').strip('!').strip('?')
                          for w in words.split() if len(w) >=4]
        all_trg_data.append((filtered_words, sentiment))

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



proc_review("negative.txt", "negReviews.txt")
proc_review("positive.txt", "posReviews.txt")

fill_data()
# word_features = nltk.FreqDist( get_just_words(all_trg_data) ).keys()
#
# print (positive_trg[1])
#
# print (extract_features(positive_trg[1]))

print (all_trg_data[1])
classifier = NaiveBayesClassifier.train(all_trg_data[1])

classifier.show_most_informative_features()