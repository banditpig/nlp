# Set up spaCy
from spacy.en import English
parser = English()

# Test Data


# Let's look at the named entities of this example:
example = "David went to the University of Bath"

parsedEx = parser(example)
for token in parsedEx:
    print(token.orth_, token.ent_type_ if token.ent_type_ != "" else "(not an entity)")

print("-------------- entities only ---------------")
# if you just want the entities and nothing else, you can do access the parsed examples "ents" property like this:
ents = list(parsedEx.ents)
for entity in ents:
    print(entity.label, entity.label_, ' '.join(t.orth_ for t in entity))

print("-------------- messy ---------------")
messyData = "lol that is rly funny :) This is gr8 i rate it 8/8!!!"
parsedData = parser(messyData)
for token in parsedData:
    print(token.orth_, token.pos_, token.lemma_)