from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import get_custom_objects
import keras
import numpy as np

SEQ_LEN = 60


#Load the model

def loadModel_vocab(path):
    vocab_path = path+'/vocab_uncased_81_71.txt'
    token_dict = load_vocabulary(vocab_path)
    tokenizer = Tokenizer(token_dict)
    
    model = keras.models.load_model(path +'/model_finalized_v1_81_71.h5', custom_objects=get_custom_objects())
    return model, tokenizer


#get prediction from loaded model
def getPredictionFromBERT(text, model, tokenizer):
    
    ids, segments = tokenizer.encode(input_text, max_len=SEQ_LEN)
    model_input = [np.array([ids]), np.zeros_like([ids])]
    prediction = model1.predict(model_input)
    
    return prediction   

input_text = "For patients who remain symptomatic on a short acting bronchodilator"
#way to load the model
model1, tokenizer1 = loadModel_vocab(path = "/home/ajeetsingh/Documents/VCC/keyMessage/BERT")

#way to get the prediction
print(getPredictionFromBERT(input_text, model1, tokenizer1)[0])
    

