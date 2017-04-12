# import modules & set up logging
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
sentences = []

# with open("posReviews.txt", "r") as f:
#     for line in f:
#         sentences.append(line.split())
# with open("negReviews.txt", "r") as f:
#     for line in f:
#         sentences.append(line.split())

with open("sandals-sat.txt", "r") as f:
    for line in f:
        sentences.append(line.split())


model = gensim.models.Word2Vec(sentences, min_count=1)

print ("ok")
print (model.most_similar(positive="shade".split(), topn=5))

# negative = "butlers".split(),