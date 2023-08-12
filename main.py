import nltk
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
import tflearn
import tensorflow as tf
import random
import json

from nltk.stem import WordNetLemmatizer

import pickle
from tensorflow.keras.models import load_model

lemmatizer=WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))

classes=pickle.load(open('classes.pkl','rb'))

model = load_model('Chat_bot.h5')
def clean_up_sen(sentences):
    sentence_words=nltk.word_tokenize(sentences)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
def bag_of_words(sentences):
    sentences_words=clean_up_sen(sentences)
    bag=[0] * len(words)

    for w in sentences_words:
        for i,word in enumerate(words):
            if word == w :
                bag[i]=1
    return np.array(bag)
def predict_class(sentences):
    bag=bag_of_words(sentences)
    res=model.predict(np.array([bag]))[0]

    ERROR_THRESHOLD=0.25

    result = [[i,r] for i ,r in enumerate(res) if r> ERROR_THRESHOLD ]

    result.sort(key=lambda x:x[1],reverse=True )
    return_list=[]
    for r in result:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list , intents_json):
    tag=intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag :
            result=random.choice(i['responses'])
            break
    return result
print("GO!")
print("Enter xxx to end")
while True:

    message=input("ENTER:")
    if(message=='xxx'):
        exit()
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)