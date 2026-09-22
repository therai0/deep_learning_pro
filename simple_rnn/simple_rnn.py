""" 
Building simple langauge model with the help of simple RNN architecture

Architecture of RNN:
--> Many to one

"""
import numpy as np 
import pandas as  pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.callbacks import EarlyStopping ,TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding


class TextPreprocessing:
    def __init__(self,file_path):
        self.file_path = file_path


    def read_file(self,file_path)->str:
        """ Read the file and return the text """
        try:
            with open(file_path,'r') as f:
                text = f.read()
                return text
        except FileNotFoundError as e:
            print(e)


    def tokenize_text(self,text):
        """ 
        Convert the text into id(integer) i.e 1,2,3,4
        """
        try:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts([text])
            tokens = tokenizer.texts_to_sequences([text])[0]
            return tokens
        except Exception as e:
            print(e)

    def create_embeddings(self,tokens, vocab_size, embedding_dim=4):
        try:
            embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
            )
            tokens = np.array(tokens)
            vectors = embedding(tokens)
            return vectors

        except Exception as e:
            print(e)

    def init_textpreprocessing(self):
        try:
            text = str(self.read_file(self.file_path))
            # print(text.split("."))
            token = self.tokenize_text(text)
            print(token)
            print(len(token))
           
    
        except Exception as e:
            print(e)


    

    
if __name__ == "__main__":
    # read_file("data.txt")
    tp = TextPreprocessing("./simple_rnn/data.txt")
    tp.init_textpreprocessing()
