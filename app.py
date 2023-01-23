from sre_parse import Tokenizer
import tokenize
from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
model = load_model('poem_generator.h5')
app = Flask(__name__,template_folder='template')
import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st

with open('robert_frost.txt',encoding="utf8") as story:
  story_data = story.read()

# print(story_data)

# data cleaning process
import re                                # Regular expressions to use sub function for replacing the useless text from the data

def clean_text(text):
  text = re.sub(r',', '', text)
  text = re.sub(r'\'', '',  text)
  text = re.sub(r'\"', '', text)
  text = re.sub(r'\(', '', text)
  text = re.sub(r'\)', '', text)
  text = re.sub(r'\n', '', text)
  text = re.sub(r'“', '', text)
  text = re.sub(r'”', '', text)
  text = re.sub(r'’', '', text)
  text = re.sub(r'\.', '', text)
  text = re.sub(r';', '', text)
  text = re.sub(r':', '', text)
  text = re.sub(r'\-', '', text)

  return text

# cleaning the data
# lower_data = story_data.lower()           # Converting the string to lower case to get uniformity

split_data = story_data.splitlines()      # Splitting the data to get every line seperately but this will give the list of uncleaned data

# print(split_data)                         

final = ''                                # initiating a argument with blank string to hold the values of final cleaned data

for line in split_data:
  line = clean_text(line)
  final += '\n' + line

# print(final)

final_data = final.split('\n')       # splitting again to get list of cleaned and splitted data ready to be processed
# print(final_data)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Instantiating the Tokenizer
max_vocab = 1000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(final_data)

# # Getting the total number of words of the data.
# word2idx = tokenizer.word_index
# # print(len(word2idx))
# # print(word2idx)
# vocab_size = len(word2idx) + 1        # Adding 1 to the vocab_size because the index starts from 1 not 0. This will make it uniform when using it further
# # print(vocab_size)

# # input_seq = []
# # for line in final_data:
# #   token_list = tokenizer.texts_to_sequences([line])[0]
# #   for i in range(1, len(token_list)):
# #     n_gram_seq = token_list[:i+1]
# #     input_seq.append(n_gram_seq)

# # print(input_seq)

# # # Getting the maximum length of sequence for padding purpose
# # max_seq_length = max(len(x) for x in input_seq)
# # # print(max_seq_length)

# # # Padding the sequences and converting them to array
# # input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_length, padding='pre'))
# # # print(input_seq)

# # # Taking xs and labels to train the model.

# # xs = input_seq[:, :-1]        # xs contains every word in sentence except the last one because we are using this value to predict the y value
# # labels = input_seq[:, -1]     # labels contains only the last word of the sentence which will help in hot encoding the y value in next step
# # # print("xs: ",xs)
# # # print("labels:",labels)

# from tensorflow.keras.utils import to_categorical

# # one-hot encoding the labels according to the vocab size

# # The matrix is square matrix of the size of vocab_size. Each row will denote a label and it will have 
# # a single +ve value(i.e 1) for that label and other values will be zero. 

# ys = to_categorical(labels, num_classes=vocab_size)
# print(ys)
def predict_words(seed, no_words):
  for i in range(no_words):
    
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences([token_list], maxlen=11-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=1)

    new_word = ''

    for word, index in tokenizer.word_index.items():
      if predicted == index:
        new_word = word
        break
    seed += " " + new_word
  return seed


@app.route('/')
def main():
    return render_template('index.html')



@app.route('/pred',methods=["POST"])
def pred():
      a = request.form['poem']
      
      next_words = 10

      res=predict_words(a, next_words)
      
      

      return render_template('index.html', data = res)
     
if __name__ == "__main__":
    app.run(debug=True)
# st.title("Poem generator")
# a, b = st.columns(2)
# inp = a.text_input('Write few words:')
# next_words = 10

# res=predict_words(a, next_words)
# result = st.button("Compose")
# st.subheader(result)
