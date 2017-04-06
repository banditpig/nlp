import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import sent_tokenize, word_tokenize

# https://www.youtube.com/watch?v=IMKweOTFjXw
positive = []
negative = []
positive_trg = []
negative_trg = []
positive_test = []
negative_test= []

all_trg_data = []

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

    positive_test = positive[:100]
    positive_trg  = positive[100:]
    negative_test = negative[:100]
    negative_trg  = negative[100:]

    for words, sentiment in positive_trg + negative_trg:
        filtered_words = [w.lower().strip('.').strip(',').strip(';').strip(';').strip(':').strip('"').strip('!').strip('?')
                          for w in words.split() if len(w) >=4]
        all_trg_data.append((filtered_words, sentiment))


proc_review("negative.txt", "negReviews.txt")
proc_review("positive.txt", "posReviews.txt")

fill_data()
print (all_trg_data)



