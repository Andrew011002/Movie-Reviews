import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize as tokenizer
from nltk.stem import WordNetLemmatizer
import os

path = os.path.dirname(os.path.abspath(__file__))

# helper function to get pos of word so that it's lemmatized correctly
def pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean(reviews):

    lemmatizer = WordNetLemmatizer() # initializing lemmatizer object

    stop_words = set(stopwords.words("english")) # create a set of stop words

    N = len(reviews) # get the number of reviews

    # apply filter to each review
    for i, review in enumerate(reviews):
        
        review = re.sub("<[^>]*>", " ", review) # stripping HTML tags
        review = re.sub("[\W]+", " ", review.lower()) # remove all non-word characters

        review = tokenizer(review) # tokenize the review
        review = [lemmatizer.lemmatize(t, pos(t)) for t in review if t not in stop_words] # strip tokens from stop words and lemmatize
        reviews[i] = " ".join(review) # join the tokens back together

        if (i + 1) % 5000 == 0:
            print(f"{i + 1} review(s) processed / {N}")
    
    print(f"{N} / {N} reviews processed")

    return reviews

def predict(text, model):
    pred = model.predict(text)
    print(f"Prediction: {'Positive Review' if pred[0] == 1 else 'Negative Review'}")

