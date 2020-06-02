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



# Defining Functions

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

# Extracting Data
words=[]
classes = []
documents = []
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((pattern, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Creating a DataFrame and shuffling
documents_df = pd.DataFrame(documents,columns=['pattern','classes'])
documents_df = documents_df.sample(frac=1,random_state=42).reset_index().drop('index',axis=1)


y = pd.get_dummies(documents_df.classes)
unique_classes = y.columns
unique_classes

# Creating Vocabulary List

words = normalize(words)
words = sorted(list(set(words)))


# Saving words and unique_classes Objects

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(unique_classes,open('classes.pkl','wb'))

#Creating training set

count_vect = CountVectorizer(analyzer = normalize,vocabulary=words)
X_train = count_vect.fit_transform(documents_df.pattern.apply(process_text)).toarray()
y_train = y.to_numpy()

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.6))
#model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)