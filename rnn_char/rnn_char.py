""" 
Implementing the RNN to predict the next word
"""
import re 
import pandas as  pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as  tf 

class CharPreprocessing:
    def __init__(self,file_path):
        self.file_path = file_path  


    def read_file(self)->str:
        try:
            with open(self.file_path,'r') as f:
                text = f.read() 
            return text
        except FileNotFoundError as e:
            print(e)


    def init_char_preprocessing(self):
        try:
            text = self.read_file().lower()
            print(len(text))
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            single_sentece = ""
            for c in clean_text:
                if c != " ":
                    single_sentece += c 

            vocabulary = sorted(set(single_sentece))
            char_id = {char:i for i,char in enumerate(vocabulary)}

            chars_to_id = [char_id[ch] for ch in single_sentece]
            
            train = chars_to_id[:800]
            test = chars_to_id[800:]

            return train,test ,vocabulary

            
        except Exception as e:
            print(e)


class BuildModel:
    def __init__(self,train,test,vocabulary):
        self.train = train 
        self.test = test 
        self.vocabulary = vocabulary 

    def build_model_train(self):
        try:
            X_train = self.train[:-1]
            y_train = self.train[1:]
            
            X_test = self.test[:-1]
            y_test = self.test[1:]
            
            X_train = tf.one_hot(X_train,depth=len(self.vocabulary))
            y_train = tf.one_hot(y_train,depth=len(self.vocabulary))
            
            X_test = tf.one_hot(X_test,depth=len(self.vocabulary))
            y_test = tf.one_hot(y_test,depth=len(self.vocabulary))
            
            # # adding batch for both train and test 
            X_train_batch = tf.expand_dims(X_train,axis=0)
            y_train_batch = tf.expand_dims(y_train,axis=0)
            
            X_test_batch = tf.expand_dims(X_test,axis=0)
            y_test_batch = tf.expand_dims(y_test,axis=0)


            model = Sequential()
            model.add(SimpleRNN(128,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(Dense(len(vocabulary),activation="softmax"))

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            early_stop = EarlyStopping(
                patience=5
            )
            model.fit(
                X_train_batch,
                y_train_batch,
                epochs=300,
                verbose=1,
                validation_data=(X_test_batch, y_test_batch),
                # callbacks=[early_stop]
            )
            model.save("model.keras")
        except Exception as e:
            print(e)




if __name__ == "__main__":
    file_path = "./rnn_char/data.txt"
    preprocessing = CharPreprocessing(file_path=file_path)
    train,test,vocabulary = preprocessing.init_char_preprocessing()
    buildmodel = BuildModel(train,test,vocabulary)
    buildmodel.build_model_train()
    