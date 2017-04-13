import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag
text = "This is a sentence. And Mr and Mrs Smith are going on holiay, in France. Mr Smith has a new shirt, his wife a new hat. Are they taking the kids? Perhaps they will, or maybe not!"

sents = sent_tokenize(text)
print (sents)

words = word_tokenize(text)
print (words)

print ("------")
print (nltk.pos_tag(words))

def entities(text):
    #  binary=True
    return ne_chunk(pos_tag(word_tokenize(text)),)

def get_nouns(text):
    nouns = []
    tagged = pos_tag(word_tokenize(text))
    for (wd,tag) in tagged:
        if (tag == 'NNP' or  tag == 'NN'):
            nouns.append(wd)

    return nouns

# text = "The recording calls into question evidence given in 2012 to the Treasury select committee by former Barclays boss Bob Diamond and Paul Tucker, the man who went on to become the deputy governor of the Bank of England."
text = "Etim went to the University of Surrey"

tree = entities(text)
tree.pprint()
print (get_nouns(text))
tree.draw()
