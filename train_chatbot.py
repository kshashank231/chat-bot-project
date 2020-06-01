import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import json
import pickle

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


import re, string, unicodedata





words=[]
classes = []
documents = []
#ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

def process_text(text):
  token_words = nltk.word_tokenize(text)
  return token_words

def remove_non_ascii(words):
  """Remove non-ASCII characters from list of tokenized words"""
  new_words = []
  for word in words:
      new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
      new_words.append(new_word)
  return new_words
def to_lowercase(words):
  """Convert all characters to lowercase from list of tokenized words"""
  new_words = []
  for word in words:
      new_word = word.lower()
      new_words.append(new_word)
  return new_words
def remove_punctuation(words):
  """Remove punctuation from list of tokenized words"""
  new_words = []
  for word in words:
      new_word = re.sub(r'[^\w\s]', '', word)
      if new_word != '':
          new_words.append(new_word)
  return new_words

def remove_stopwords(words):
  """Remove stop words from list of tokenized words"""
  new_words = []
  for word in words:
      if word not in stopwords.words('english'):
          new_words.append(word)
  return new_words

def stem_words(words):
  """Stem words in list of tokenized words"""
  stemmer = LancasterStemmer()
  stems = []
  for word in words:
      stem = stemmer.stem(word)
      stems.append(stem)
  return stems

def lemmatize_verbs(words):
  """Lemmatize verbs in list of tokenized words"""
  lemmatizer = WordNetLemmatizer()
  lemmas = []
  for word in words:
      lemma = lemmatizer.lemmatize(word, pos='v')
      lemmas.append(lemma)
  return lemmas

def normalize(words):

  words = remove_non_ascii(words)
  words = to_lowercase(words)
  words = remove_punctuation(words)
  words = remove_stopwords(words)
  
  words = lemmatize_verbs(words)
  return words

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents[0])
print(documents[1])
print(documents[2])
# lemmaztize and lower each word and remove duplicates
#words = normalize(words)
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)


documents_df = pd.DataFrame(documents,columns=['pattern','classes'])
y_train = pd.get_dummies(documents_df.classes)
count_vect = CountVectorizer(analyzer=normalize)
words_count = count_vect.fit(words)
X_train =  count_vect.transform(documents_df.pattern)
print(X_train.shape)
# print(type(bag))
# print(bag.shape)
# print('-----------------')
# # training set, bag of words for each sentence
# for doc in documents:
#     # initialize our bag of words
#     #bag = []
#     # list of tokenized words for the pattern
#    #pattern_words = doc[0]
#     # lemmatize each word - create base word, in attempt to represent related words
#     #pattern_words = normalize(words)

#     # count_vect = CountVectorizer(analyzer=normalize)
#     # words_count = count_vect.fit(words)
#     # bag =  count_vect.transform(pattern_words)

#     # create our bag of words array with 1, if word match found in current pattern
#     # for w in words:
#     #     bag.append(1) if w in pattern_words else bag.append(0)
    
#     # output is a '0' for each tag and '1' for current tag (for each pattern)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
    
#     training.append([bag[doc], output_row])
# # shuffle our features and turn into np.array
# #print(training[0])
# print(bag)
# print(bag.toarray())
# print(training[0])
# print(type(training))
# print('-----------------------------------------')
# random.shuffle(training)
# training = np.array(training)
# print(training[0])
# print(type(training))
# # create train and test lists. X - patterns, Y - intents
# train_x = list(training[:,0])
# train_y = list(training[:,1])
# print("Training data created")


# # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # equal to number of intents to predict output intent with softmax
# # model = Sequential()
# # model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# # model.add(Dropout(0.6))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(len(train_y[0]), activation='softmax'))

# # # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # #fitting and saving the model 
# # hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# # model.save('chatbot_model.h5', hist)

# # print("model created")
