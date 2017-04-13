import pickle
import os.path
import nltk
import csv

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from random import shuffle

from nltk.tokenize import sent_tokenize

class NLP:
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

    def proc_review(self, fName, outName):
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


    def fill_data(self):

        with open("posReviews.txt", "r") as f:
            for line in f:
                NLP.positive.append((line, NLP.POSITIVE))
        with open("negReviews.txt", "r") as f:
            for line in f:
                NLP.negative.append((line, NLP.NEGATIVE))

        NLP.positive_test = NLP.positive[:200]
        NLP.positive_trg = NLP.positive[200:]
        NLP.negative_test = NLP.negative[:200]
        NLP.negative_trg = NLP.negative[200:]

        with open("positive_test.txt", "w") as pos_file:
            for line in NLP.positive_test:
                (txt, _) = line
                pos_file.write(txt)

        with open("negative_test.txt", "w") as pos_file:
            for line in NLP.negative_test:
                (txt, _) = line
                pos_file.write(txt)



                        # global all_trg_data
        # all_trg_data = positive_trg + negative_trg
        # shuffle(all_trg_data)
        for words, sentiment in NLP.positive_trg + NLP.negative_trg:
            filtered_words = [w.lower().strip('.').strip(',').strip(';').strip(';').strip(':').strip('"').strip('!').strip('?')
                              for w in words.split() if len(w) >=4]
            NLP.all_trg_data.append((filtered_words, sentiment))
        shuffle(NLP.all_trg_data)

    def get_just_words(data):
        all_words = []
        for(words, sentiment) in data:
            all_words += words
        return all_words;


    def extract_features(src_doc):
        document_words = set(src_doc)
        features = {}

        for word in NLP.word_features:
            features ['contains(%s)' % word] =  (word in document_words)

        return features




    def get_classifier_probabilities(self, text):
        prob_dist = classifier.prob_classify(text)
        return prob_dist.max(), round(prob_dist.prob(NLP.POSITIVE), 2), round(prob_dist.prob(NLP.NEGATIVE), 2)


    def dump_results(self):
        neg = 0
        pos = 0
        print("Testing NEGATIVES")
        for (review, _) in NLP.negative_test:
            if classifier.classify(review) == NLP.POSITIVE:
                pos = pos + 1
            else:
                neg = neg + 1

            print(classifier.classify(review) + " pos " + str(pos) + " neg " + str(neg))

        print("------------------------------------------")
        print("Testing POSITIVES")

        neg = 0
        pos = 0
        for (review, _) in NLP.positive_test:
            if classifier.classify(review) == NLP.NEGATIVE:
                neg = neg + 1
            else:
                pos = pos + 1

            print(classifier.classify(review) + " pos " + str(pos) + " neg " + str(neg))
    def get_classifier(self, trg_set):

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


    def get_sentiment_file_of_reviews(self, fname, outname):
        with open(fname, "r") as f:

            out_file = open(outname , 'w')
            writer = csv.writer(out_file)


            for line in f:
                    print (line)
                    res = self.get_classifier_probabilities( line)
                    print (res)
                    print ("")
                    writer.writerow((line.strip('\n\r').strip('"'),) + res)

            out_file.close()

    def get_sentences(self, text):
        return sent_tokenize(text)

    def top_sentences(self, text):
        print ("---- Review text. ----")
        print (text)
        sentences = self.get_sentences( text)
        print ("---- sentences and scores...")
        for sen in sentences:
            (clas, pscore, nscore) = self.get_classifier_probabilities(sen)
            print (sen + " -> " + clas + " + " + str(pscore) + " - " + str(nscore))
            # print (res)
        print("=======================")
        print ("")


nlp = NLP()
# nlp.proc_review("negative.txt", "negReviews.txt")
# nlp.proc_review("positive.txt", "posReviews.txt")
nlp.fill_data()

# for (_, s) in all_trg_data[:100]:
#     print (s)
classifier = nlp.get_classifier(nlp.all_trg_data[:2800])

# print (classifier.accuracy(NLP.positive_test))
# print (classifier.accuracy(NLP.negative_test))

# print (nlp.get_classifier_probabilities("I purchased this for my daughter a bit before Christmas and she loves it.  It is a great item for the price and does what the bigger brand names does.  Thank you."))
# print (nlp.get_classifier_probabilities("This was excellent, very useful. Get one!"))
#
# print (nlp.get_classifier_probabilities("This is awful. Dont bother, piece of rubbish"))
#
# print (nlp.get_classifier_probabilities("No sure, maybe ok, could be better."))
# print (nlp.get_classifier_probabilities("you must be joking, this is terrible."))

# nlp.get_sentiment_file_of_reviews("positive_test.txt", "positive_test_results.csv")
# nlp.get_sentiment_file_of_reviews("negative_test.txt", "negqtive_test_results.csv")
#
# nlp.get_sentiment_file_of_reviews("sandals-sat.txt", "sandals.csv")

# nlp.dump_results()

# Accuracy: 0.984

# print("Accuracy: {0}".format(classifier.accuracy(NLP.all_trg_data)))

# NEGATIVE pos 33 neg 67 1500
# POSITIVE pos 79 neg 21
#
# NEGATIVE pos 39 neg 61  2000
# POSITIVE pos 86 neg 14
# classifier.show_informative_features(10)

sentisumText = ["Exactly what I expected.",
"Excellent",
"Excellent",
"Great product! Love it! Would recommend.",
"Good",
"Excellent quality but why the shirt has to be that long?",
"Product as expected",
"Lovely and comfy, neat look",
"Really nice shirt, probably going to take advantage of the 4 for £100 deal very soon to get four more!",
"Poor",
"No issues absolutely. Good product, great price when the sale was on",
"Average",
"No problems",
"Great quality, fit and price.",
"Great product delivered quickly with great packaging.",
"Great fitting shirt will look to order more",
"Good fit and nice comfortable material.",
"Can you make a bigger range for people with 38 inch arm length.The choice is very poor",
"Good quality as always",
"Good quality shirt, fits well. Should have tried it in store cause I think it shines a bit too much. (Hard to tell by the given description, I think this is the one with the regular cuffs)",
"Beautiful shirt, thick but not hot.",
"Great fit glad I found out about slim fit. I'm a little porky and regular shirt are like a tent . The only disappointing thing is the shortage of short sleeved shirts in your range . A couple of colours and styles ,I think would a winner",
"Good quality shirts at a fair price."]

for txt in sentisumText:
    nlp.top_sentences(txt)
