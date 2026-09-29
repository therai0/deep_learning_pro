""" 
Building the next word prediction model using LSTM 

1) Read the data from nltk (shakespeare-hamlet.txt)
2) Preprocessing and tokenization
==> First to convert the word into numeric value 
==> Convert into input sequence 
==> train and test 

3) Build the standard LSTM model and embedding 
4) Predicting the next word
"""

import nltk 
# nltk.download("gutenberg")
from nltk.corpus import gutenberg
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences,to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import SparseCategoricalCrossentropy,CategoricalCrossentropy
from tensorflow.keras.layers import  Input,Dense,Dropout,Embedding ,LSTM

from sklearn.model_selection import  train_test_split


""" 
Read the gutenberg  shakespeare-hamlet.txt data and save in data.txt file 
"""
def downloadData():
    try:
        data = gutenberg.raw('shakespeare-hamlet.txt').lower()
        with open("./lstm_gru/data.txt",'w') as file:
            file.write(data)
    except Exception as e:
        print(e)


class TextPreprocessing:
    """ 
    After complete preprocessing return the X,y 
    """
    def __init__(self,file_path):
        self.file_path = file_path 


    def read_data(self):
        """ read the data from the particular file path """
        try:
            with open(self.file_path,'r') as file:
                file = file.read()
                return file 
        except Exception as e:
            print(e) 

    def tokenization(self,text):
        """ fitting to text 
            return tokenizer 
        """
        try:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts([text])
            return tokenizer 
        except Exception as e:
            print(e)
    

    def input_sequence(self,tokenizer,text)->list:
        """ 
        Convert the text into token 
        and return in list 
        """
        try:
            input_sequence_lst = []
            for line in text.split("\n"):
                token = tokenizer.texts_to_sequences([line])[0]
                for i in range(1,len(token)):
                    input_sequence_lst.append(token[:i+1])
            return input_sequence_lst
        except Exception as e:
            print(e)


    def padding_sequence(self,input_sequence_lst,max_len):
        """ 
        convert the tokens into equal number in lenght by adding extra padding 
        padding style is pre
        """
        try:
            pad_input_sequence = pad_sequences(input_sequence_lst,maxlen=max_len)
            return pad_input_sequence
        except Exception as e:
            print(e)

    
    def init_preprocessing(self):
        try:
            file = self.read_data()
            tokenizer = self.tokenization(file)
            input_sequence = self.input_sequence(tokenizer,file) 
            max_len = max([len(x) for x in input_sequence])
            pad_input_sequence = self.padding_sequence(input_sequence,max_len)

            X,y = pad_input_sequence[:,:-1],pad_input_sequence[:,-1]
            y = to_categorical(y,num_classes=len(tokenizer.word_index) + 1)
            return X,y ,len(tokenizer.word_index)+1,max_len

        except Exception as e:
            print(e)


class NextWordPredictionModel:
    def __init__(self,X,y,total_word,max_seq):
        self.X = X 
        self.y = y 
        self.total_word = total_word
        self.max_seq = max_seq


    def build_model(self):
        try:
            optimizer = Adam()
            loss = CategoricalCrossentropy()
            model = Sequential()
            model.add(Input(shape=(self.max_seq - 1,)))
            model.add(Embedding(self.total_word,100))
            model.add(LSTM(150,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dense(self.total_word,activation="softmax"))
            model.compile(loss=loss,optimizer=optimizer,metrics =["accuracy"])
            return model 
        except Exception as e:
            print(e) 

    def train_model(self):
        try:
            X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.2)
            model = self.build_model()

            train_model = model.fit(X_train,y_train,epochs= 50,validation_data=(X_test,y_test),verbose=1) 
            model.save("next_word_prediction_model.keras")
            
        except Exception as e:
            print(e)

if __name__ == "__main__":
    # downloadData()
    tp = TextPreprocessing("./lstm_gru/data.txt")
    X,y,total_word,max_len = tp.init_preprocessing()
    
    nwpm = NextWordPredictionModel(X,y,total_word=total_word,max_seq=max_len)
    nwpm.train_model()





