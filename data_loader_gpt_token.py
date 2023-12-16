import numpy as np
import pandas as pd

import torch

from datasets import load_dataset

from transformers import GPT2Tokenizer

np.random.seed(9)


    
""" Returns GPT2 tokenizer after adding special tokens """ 
def add_special_tokens(gpt2path):
	tokenizer = GPT2Tokenizer.from_pretrained(gpt2path)
	special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
	tokenizer.add_special_tokens(special_tokens)
	return tokenizer




def data_processor(tokenizer, no_train=100, no_val=100, max_length=1000):
  
  
  """Convert the datasets to lengths and drops large lengths"""
  
  dataset = load_dataset("cnn_dailymail", "3.0.0")
  
  """train valid split (90:10)"""
  train_indices = np.random.choice(len(dataset["train"]), no_train, replace=False)
  val_indices = np.random.choice(len(dataset["validation"]), no_val, replace=False)
  
  traindata = dataset["train"].select(train_indices)
  valdata = dataset["validation"].select(val_indices)
  
  #Adding length of the text to the dataset object using map function
  def add_length(data_set):
    data_set['length'] = len(tokenizer(data_set['article'])['input_ids']) + len(tokenizer(data_set['highlights'])['input_ids'])
    data_set['text_length'] = len(tokenizer(data_set['article'])['input_ids'])
    return data_set
  
  traindata = traindata.map(add_length)
  valdata = valdata.map(add_length)
  
  """ removing items exceeding the max length limit"""
  traindata = traindata.filter(lambda x: x['length'] <= max_length-10)
  #traindata.reset_index(drop=True, inplace=True)
  valdata = valdata.filter(lambda x: x['length'] <= max_length-10)
  #valdata.reset_index(drop=True, inplace=True)
  
  return traindata, valdata
  
  

""" Preparing data batch for GPT-2"""

""" We are setting batch size as 1 due to gpu memory limitation and will use gradient accmulation. 
Hence we will not pad the text in order to make the training procees more efficient"""

def smart_batching(data, tokenizer, max_length=1000):
  #data is a higgings dataset object with article and highlights columns

  databatch = []
  for index, row in enumerate(data):

    article = row['article']
    summary = row['highlights']
    
    text = '<|startoftext|> ' + article + ' <|summarize|> '+ summary + ' <|endofthetext|>'
    input_id = tokenizer(text, padding='max_length', max_length=1000, truncation=True)['input_ids']
    

    input_id = torch.tensor(input_id)

    #databatch.append((input_id, type_id, lm_label))
    databatch.append((input_id, text))

  return databatch 
        
       
"""Custom function to return the tokens corresponding to the unk tokens"""
def oov(input_tokens, input_data, tokenizer):
  """
  This converts the unk tokens to new tokens and returns the new tokens
  input_tokens: torch tensor of input tokens, input_data: input text
  """
    
  unk_token = tokenizer.unk_token
  unk_token_id = tokenizer.unk_token_id
  
  #Find the locations of the unk tokens
  locations = torch.nonzero(input_tokens == unk_token_id)
  #print(locations)
  #Assume the first word is always not unk
  
  unk_tokens = []
  
  prev_loc = 0
  for loc in locations:
      
    if loc == 0:
        #if its the first location
        next_token = input_tokens[loc + 1]
        decoded_next_token = tokenizer.decode(next_token)
        next_token_index = input_data.find(decoded_next_token)
        unk_tokens.append(input_data[:next_token_index])
        input_data = input_data[next_token_index:]
        continue
    
    elif loc == len(input_tokens) - 1:
        #if its the last location
        sub_str = input_tokens[ : loc]
        decoded_sub_str = tokenizer.decode(sub_str)
        input_data = input_data[len(decoded_sub_str):]
        unk_tokens.append(input_data)
        continue
    
    else:
        
        #find the text upto the unknown token
        sub_str = input_tokens[prev_loc: loc]
        #decode it to text
        decoded_sub_str = tokenizer.decode(sub_str)
        #remove the deocded text from the input data
        input_data = input_data[len(decoded_sub_str):]
        #find the next token after the unk token
        next_token = input_tokens[loc + 1]
        #decode the next token
        decoded_next_token = tokenizer.decode(next_token)
        #find the index of the next token in the input data
        next_token_index = input_data.find(decoded_next_token)
        #find the unk token in the input data
        unk_tokens.append(input_data[:next_token_index])
        #remove the unk token from the input data
        input_data = input_data[next_token_index:]
        prev_loc = loc + 1

  #To replace the unk tokens with new temporary tokens
  
  new_tokens = list(set(unk_tokens))
  vocab_size = len(tokenizer)
  #print(input_tokens)
  #print(locations)
  
  for i in range(len(unk_tokens)):
    input_tokens[locations[i]] = vocab_size + new_tokens.index(unk_tokens[i])
  
  #unkkens are the tokens that are unknown with repetition new tokens are set of this
  #input tokens are the updated tokens with new tokens with temp ids 
            
  #return unk_tokens, input_tokens, new_tokens
  
  return unk_tokens, new_tokens
  
  #return unk_tokens
