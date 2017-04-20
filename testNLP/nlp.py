import pickle
import os.path
import nltk
import csv
import jsonpickle
import json

import pymongo
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from random import shuffle

from nltk.tokenize import sent_tokenize
from pymongo import MongoClient
from pymongo.son_manipulator import SONManipulator
import collections  # From Python standard library.
import bson
from bson.codec_options import CodecOptions
from bson import Binary, Code, ObjectId
from bson.json_util import dumps


def jdefault(o):
    return o.__dict__



class Sentence_Result((object)):
    def __init__(self):

        self.text = ""
        self.sentiment = 0.0
        self.features= []




class Result(object):
    def __init__(self):
        self.NLP_SERVICE = "NLTK"
        self.id = ""
        self.merchant = ""
        self.text = ""
        self.sentiment = 0.0
        self.features = []
        self.sentences = []
        self._id = ""
#
# res = Result()
# res.review = "hello"
# res.review_score = 0.8
# res.features = ["one", "two", "three"]
# sc1 = Sentence_Score()
# sc1.sentence = "sentence one"
# sc1.score = 0.1
#
# sc2 = Sentence_Score()
# sc2.sentence = "another one"
# sc2.score = 0.1
#
# res.feature_sentences = {'one' : ["dghsdds one dsdsd", "fff one one fsf"]}
# res.feature_sentences['two'] = ["dghsdds two dsdsd", "fff two two fsf"]
# # [sc1, sc2]
# # res.feature_sentences = [("one", [sc1, sc2])]
#
# print (json.dumps(res, indent=4, default=jdefault))

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
                if ("\"id\":" in  line) or ("id:" in  line):
                    id = line.split(":")[1]
                    #     skip one line until find review


                    reviewLine = next(f)
                    while "\"review\":" not in reviewLine:
                        reviewLine = next(f)

                    review = reviewLine.split(":")[1]
                    outfile.writelines(id.replace('\\n', ' ').replace('"','').strip('\n\r').strip(',').strip() + "," + review.strip().replace('\\n', ' ').replace('"','').strip('\n\r').strip(','))
                    outfile.writelines('\n')


        outfile.close()


    def review_lines(self, fname):
        list_lines = []
        with open(fname, "r") as f:
            for line in f:
                list_lines.append(line)

        return list_lines

    def process_reviews_file(self, fname, merchant, outfile):


        lines = self.review_lines(fname)
        for fullLine  in lines[:10]:

            [id, line] = fullLine.split(',', 1)
            line = line.replace("\"", "").replace("Â£"," ").replace("âºâºâºâº"," ")
            line = line.lower()
            res = Result()
            res.text = line
            res.id = id
            (clas, pscore, nscore) = self.get_classifier_probabilities(line)
            res.sentiment = str(pscore)
            res.merchant = merchant

            noun_phrases = self.get_noun_phrases(line)
            if len(noun_phrases) == 0:
                noun_phrases = self.get_nouns(line)

            res.features = noun_phrases

            # eac h line is a review and could be many sentences
            sentences = self.get_sentences(line)


            sentence_results = []
            for sentence in sentences:
                sentence_result = Sentence_Result()
                sentence_result.text = sentence
                (clas, pscore, nscore) = self.get_classifier_probabilities(sentence)
                the_score = str(pscore)
                sentence_result.sentiment = the_score
                # get features
                sentence_features = []

                for feature in noun_phrases:
                    if feature in sentence:
                        sentence_features.append(feature)


                sentence_result.features = sentence_features
                sentence_results.append(sentence_result)

            res.sentences = sentence_results

            res._id = str(ObjectId())
            doc_str = json.dumps(res, indent=4, default=jdefault) #
            print(doc_str)

            outfile.write(doc_str)

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

    def get_just_words(self, data):
        all_words = []
        for(words, sentiment) in data:
            all_words += words
        return all_words;


    def extract_features(self, src_doc):
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

    def get_nouns(self, text):
        is_noun = lambda pos: pos[:2] == 'NN'
        # do the nlp stuff
        tokenized = nltk.word_tokenize(text)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

        return nouns

    def get_noun_phrases(selfself, text):
        blob = TextBlob(text)
        noun_phrases = blob.noun_phrases

        return noun_phrases

    def sentences_with_noun(self, noun, sentences): #feature is really a noun

        unique_sentences = []
        for sentence in sentences:
            if noun in sentence:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)

        # print(unique_sentences)
        # print("=======================")
        return unique_sentences


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


client = MongoClient()
db = client['feefo']
results_collection = db['results']
#
# res = Result()
# res._id = str(ObjectId())
# res.features = ["dd", "aa"]
# res.sentences = ["dsddddsdsdsd"]
# doc_str = dumps(res, indent=4, default=jdefault)  #
# print(doc_str)
# # results_collection.insert_one(res)
# results_collection.save(dict(res))

nlp = NLP()

# text = ("Great fit glad I found out about slim fit. I'm a little porky and regular shirt are like a tent . The only disappointing thing is the shortage of short sleeved shirts in your range . A couple of colours and styles ,I think would a winner")
# text = " ".join(sentisumText)
# text = "Love these shirts, great fit and typical TM Lewin high quality"
#
# sentences = nlp.get_sentences(text)
# nouns = nlp.get_nouns(text)
# noun_phrases = nlp.get_noun_phrases(text)
# if len(noun_phrases) == 0:
#     noun_phrases = nlp.get_nouns(text)
# for noun in noun_phrases:
#     sens = nlp.sentences_with_noun(noun,sentences)


# nlp.proc_review("negative.txt", "negReviews.txt")
# nlp.proc_review("positive.txt", "posReviews.txt")
nlp.fill_data()

# for (_, s) in all_trg_data[:100]:
#     print (s)
classifier = nlp.get_classifier(nlp.all_trg_data[:2800])

out_file = open("results.json", 'w')
all_data = [("Axa.json", "axa.txt","Axa"), ("sandals.json", "sandals.txt", "Sandals"), ("tmlewin.json", "tmlewin.txt", "TM Lewin")]
for (json_file, reviews_out_file, merchant_name) in all_data:
  nlp.proc_review(json_file, reviews_out_file)
  nlp.process_reviews_file( reviews_out_file, merchant_name, out_file)

out_file.close()
print ("done")

# classifier.show_informative_features(20)

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


#
# for txt in sentisumText:
#     nlp.top_sentences(txt)
