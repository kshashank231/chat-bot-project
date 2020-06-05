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
from keras.models import load_model
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
    #words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    
    return words

# Extracting Data
words=[]
classes = []
all_classes = []
context = []
documents = []
data_file = open('intents.json').read()
intents = json.loads(data_file)

try:
  with open('data.pkl','rb') as f:
    words,classes,all_classes,context_dict = pickle.load(f)
  model = load_model('chatbot_model.h5') 

except:
  for intent in intents['intents']:
    all_classes.append(intent['tag'])
    
    context.extend(intent['context'])
    
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((pattern, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
  context_dict = dict(zip(all_classes,context))
    # for intent in intents['intents']:
    #     for pattern in intent['patterns']:

    #         #tokenize each word
    #         w = nltk.word_tokenize(pattern)
    #         words.extend(w)
    #         #add documents in the corpus
    #         documents.append((pattern, intent['tag']))

    #         # add to our classes list
    #         if intent['tag'] not in classes:
    #             classes.append(intent['tag'])

    # Creating a DataFrame and shuffling
  documents_df = pd.DataFrame(documents,columns=['pattern','classes'])
  documents_df = documents_df.sample(frac=1,random_state=42).reset_index().drop('index',axis=1)


  y = pd.get_dummies(documents_df.classes)
  classes = y.columns
    
    # Creating Vocabulary List

  words = normalize(words)
  words = sorted(list(set(words)))


    # Saving words and unique_classes Objects

  with open('data.pkl','wb') as f:
    pickle.dump((words,classes,all_classes,context_dict),f)

    #Creating training set

  count_vect = CountVectorizer(analyzer = normalize,vocabulary=words)
  X_train = count_vect.fit_transform(documents_df.pattern.apply(process_text)).toarray()
  y_train = y.to_numpy()

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
  model = Sequential()
  model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
  model.add(Dropout(0.3))
    #model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(len(y_train[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #fitting and saving the model 
  hist = model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
  model.save('chatbot_model.h5', hist)




def clean_up_sentence(sentence):
    words = process_text(sentence)
    return normalize(words)

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):    
    sentence_words = clean_up_sentence(sentence)
    count = CountVectorizer(vocabulary=words)
    bag = count.transform(sentence_words).toarray()  
    return bag[0]

def predict_class(sentence, model,classes = classes,all_classes=all_classes,context_dict=context_dict):

    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    pred_proba = res.max()
    ACCURACY = 0.7
    if pred_proba > ACCURACY:
        output_class = classes[res.argmax()]
        output_context = context_dict[output_class]
        
    else:
        output_class = all_classes[3]
        output_context = context_dict[output_class]

    return output_class,output_context



def getResponse(tag, intents_json):
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg,model=model):
    tag,c = predict_class(msg, model)
    res = getResponse(tag, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("ReVA-Revildy Virtual Assistant")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="10", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=10, y=401, height=90, width=265)
SendButton.place(x=270, y=401, height=90)

base.mainloop()