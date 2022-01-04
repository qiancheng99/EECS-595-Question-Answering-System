import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
from collections import Counter
from transformers import XLNetTokenizer, XLNetModel
from scipy import spatial
from sentence_transformers import SentenceTransformer, util
import nltk.data
from sentence_transformers import SentenceTransformer,util
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os.path
from os import path
from torchsummary import summary
nltk.download('punkt')

def create_examples(input_data, set_type):
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        if len(qa["answers"]) == 0:
                            print("empty answer!!!")
                            continue
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]
                # if is_impossible:
                #     print(qas_id)
                example = {"qas_id":qas_id,
                    "question_text":question_text,
                    "context_text":context_text,
                    "is_impossible":is_impossible}
            

                examples.append(example)
    return examples
    
def get_train_examples( data_dir, filename=None):
    """
    Returns the evaluation example from the data directory.
    Args:
        data_dir: Directory containing the data files used for training and evaluating.
        filename: None by default, specify this if the evaluation file has a different name than the original one
            which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
    """
    if data_dir is None:
        data_dir = ""

    with open(
        os.path.join(data_dir, filename), "r", encoding="utf-8"
    ) as reader:
        input_data = json.load(reader)["data"]
    return create_examples(input_data, "train")
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_train_emb():
  if path.exists("embeddings")&path.exists("target"):
    embeddings = pickle.load( open( "embeddings", "rb" ) )
    target = pickle.load( open( "target", "rb" ) )
  else:
    examples=get_train_examples( data_dir="", filename="train-v2.0.json")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings=[]
    target=[]
    for i in tqdm(range(len(examples))):
      context=examples[i]["context_text"]+examples[i]["question_text"]
      embedding = model.encode(context, convert_to_tensor=True)
      embeddings.append(embedding)
      if examples[i]["is_impossible"]==False:
        target.append(1) # has answer
      else:
        target.append(0) # no answer
    pickle.dump( embeddings, open( "embeddings", "wb" ) )
    pickle.dump( target, open( "target", "wb" ) )
    embeddings = pickle.load( open( "embeddings", "rb" ) )
    target = pickle.load( open( "target", "rb" ) )
  return (embeddings,target)
  
def create_train_loader(embeddings,target): 
  BATCH_SIZE = 64
  X=torch.stack(embeddings)
  Y=torch.FloatTensor(target)
  scaler = StandardScaler()
  X = scaler.fit_transform(X.cpu())
  train_data = TrainData(torch.FloatTensor(X),torch.FloatTensor(Y))
  train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
  return train_loader

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(768, 128) 
        self.layer_3 = nn.Linear(128, 32)
        self.layer_out = nn.Linear(32, 1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = torch.sigmoid(self.layer_3(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def train_stage1(train_loader):
  EPOCHS = 35
  BATCH_SIZE = 64
  LEARNING_RATE = 0.0022
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  model = BinaryClassification()
  model.to(device)
  print(model)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  model.train()
  for e in range(EPOCHS):
      epoch_loss = 0
      epoch_acc = 0
      for X_batch, y_batch in train_loader:
          X_batch, y_batch = X_batch.to(device), y_batch.to(device)
          optimizer.zero_grad()          
          y_pred = model(X_batch)          
          loss = criterion(y_pred, y_batch.unsqueeze(1))
          acc = binary_acc(y_pred, y_batch.unsqueeze(1))         
          loss.backward()
          optimizer.step()  
          epoch_loss += loss.item()
          epoch_acc += acc.item()
      print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
  torch.save(model, "stage1_model")
  return (model,device)

class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
def get_dev_emb():
  if path.exists("embeddings_dev")&path.exists("target_dev"):
    embeddings = pickle.load( open( "embeddings_dev", "rb" ) )
    target = pickle.load( open( "target_dev", "rb" ) )
    id = pickle.load( open( "id_dev", "rb" ) )
  else:
    examples=get_dev_examples( data_dir="", filename="dev-v2.0.json")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings=[]
    target=[]
    id=[]
    for i in tqdm(range(len(examples))):
      context=examples[i]["context_text"]+examples[i]["question_text"]
      embedding = model.encode(context, convert_to_tensor=True)
      embeddings.append(embedding)
      qid=examples[i]['qas_id']
      id.append(qid)
      if examples[i]["is_impossible"]==False:
        target.append(1) # has answer
      else:
        target.append(0) # no answer
    pickle.dump( embeddings, open( "embeddings_dev", "wb" ) )
    pickle.dump( target, open( "target_dev", "wb" ) )
    pickle.dump( id, open( "id_dev", "wb" ) )
    embeddings = pickle.load( open( "embeddings_dev", "rb" ) )
    target = pickle.load( open( "target_dev", "rb" ) )
    id = pickle.load( open( "id_dev", "rb" ) )
  return (embeddings,target,id)

def stage1_eval(model,embeddings,target):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  X=torch.stack(embeddings)
  Y=torch.FloatTensor(target)
  scaler = StandardScaler()
  X = scaler.fit_transform(X.cpu()) 
  test_data = TestData(torch.FloatTensor(X))
  test_loader = DataLoader(dataset=test_data, batch_size=1)
  y_pred_list = []
  model.eval()
  with torch.no_grad():
      for X_batch in test_loader:
          X_batch = X_batch.to(device)
          y_test_pred = model(X_batch)
          y_test_pred = torch.sigmoid(y_test_pred)
          # y_pred_tag = torch.round(y_test_pred)
          y_pred_list.append(y_test_pred.cpu().numpy())
  return y_pred_list

def do_stage1(device):
  if path.exists("stage1_model"):
    model = torch.load("stage1_model")
    model.to(device)
  else:
    embeddings,target=get_train_emb()
    train_loader=create_train_loader(embeddings,target)
    model,device=train_stage1(train_loader)
  print("Successfully load the model")
  emb_dev,tar_dev,id_dev=get_dev_emb()
  y_pred=stage1_eval(model,emb_dev,tar_dev)
  score={}
  for i in range(len(y_pred)):
    score[id_dev[i]]=y_pred[i].item()
  return score

def get_dev_examples( data_dir, filename=None):
    """
    Returns the evaluation example from the data directory.
    Args:
        data_dir: Directory containing the data files used for training and evaluating.
        filename: None by default, specify this if the evaluation file has a different name than the original one
            which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
    """
    if data_dir is None:
        data_dir = ""

    with open(
        os.path.join(data_dir, filename), "r", encoding="utf-8"
    ) as reader:
        input_data = json.load(reader)["data"]
    return create_examples_dev(input_data, "dev")

def create_examples_dev(input_data, set_type):
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        if len(qa["answers"]) == 0:
                            print("empty answer!!!")
                            continue
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]
                # if is_impossible:
                #     print(qas_id)
                example = {"qas_id":qas_id,
                    "question_text":question_text,
                    "context_text":context_text,
                    "is_impossible":is_impossible}
            

                examples.append(example)
    return examples

def get_dev_QA(model_name):
  name=model_name.replace('/',"")
  if path.exists("qid_"+name)&path.exists("result_"+name):
    result = pickle.load( open( "result_"+name, "rb" ) )
    qid = pickle.load( open( "qid_"+name, "rb" ) )
  else:
    examples=get_dev_examples( data_dir="", filename="dev-v2.0.json")
    # model_name = "deepset/roberta-base-squad2"
    encodingmodel = SentenceTransformer('all-mpnet-base-v2')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    result=[]
    qid=[]
    for i in tqdm(range(len(examples))):
      id=examples[i]["qas_id"]
      is_impossible=examples[i]["is_impossible"]

      context=examples[i]["context_text"]
      text = tokenizer.tokenize(context)
      q_text=examples[i]['question_text']
      embeddings1 = encodingmodel.encode(text, convert_to_tensor=True)
      embeddings2 = encodingmodel.encode(q_text, convert_to_tensor=True)

      cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
      index=torch.argmax(cosine_scores)
      if index==0:
        start=0
      else:
        start=index.item()-1
      if index==len(text):
        end=len(text)
      else:
        end=index.item()+1
      # print(start,end)
      text=text[start:end+1]
      final=""
      for j in range(len(text)):
        final=final+" "+text[j]
      QA_input = {
        'question': q_text,
        'context': final}
      res=nlp(QA_input)
      result.append(res)
      qid.append(id)
      pickle.dump( qid, open( "qid_"+name, "wb" ) )
      pickle.dump( result, open( "result_"+name, "wb" ) )
  return (result,qid)

def generate_pred(qid,result):
  pred_file="{"
  for i in range(len(qid)):
    pred_file=pred_file+"\""+qid[i]+"\": "
    if result[i]["score"]>0.19:
      pred_file=pred_file+"\""+result[i]["answer"].replace('"',"\\\"").replace('\n',"")+"\""
    else:
      pred_file=pred_file+"\"\""
    if i!=len(qid)-1:
      pred_file+=", "
  pred_file+="}"
  text_file = open("pred.json", "w")
  n = text_file.write(pred_file)
  text_file.close()


def run_shrinked_range(model_name):
  result,qid = get_dev_QA(model_name)
  generate_pred(qid,result)

def get_dev_QA_whole(model_name):
  name=model_name.replace('/',"")
  if path.exists("qid_whole_"+name)&path.exists("result_whole_"+name):
    result = pickle.load( open( "result_whole_"+name, "rb" ) )
    qid = pickle.load( open( "qid_whole_"+name, "rb" ) )
  else:
    # model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    result=[]
    qid=[]
    examples=get_dev_examples( data_dir="", filename="dev-v2.0.json")
    for i in tqdm(range(len(examples))):
      id=examples[i]["qas_id"]

      context=examples[i]["context_text"]
      q_text=examples[i]['question_text']

      QA_input = {
        'question': q_text,
        'context': context}
      res=nlp(QA_input)
      result.append(res)
      qid.append(id)
    pickle.dump( qid, open( "qid_whole_"+name, "wb" ) )
    pickle.dump( result, open( "result_whole_"+name, "wb" ) )
  return (result,qid)

def run_whole_range(model_name):
  result_whole, qid_whole = get_dev_QA_whole(model_name)
  generate_pred(qid_whole,result_whole)

def combine_2_stages(qid,result,score):
  pred_file="{"
  for i in range(len(qid)):
    score_stage1=score[qid[i]]
    score_stage2=result[i]["score"]
    final_score=0*score_stage1+1*score_stage2
    pred_file=pred_file+"\""+qid[i]+"\": "
    if final_score>0.30:
      pred_file=pred_file+"\""+result[i]["answer"].replace('"',"\\\"").replace('\n',"")+"\""
    else:
      pred_file=pred_file+"\"\""
    if i!=len(qid)-1:
      pred_file+=", "

  pred_file+="}"
  text_file = open("pred_with2stage.json", "w")
  n = text_file.write(pred_file)
  text_file.close()

def only_stage2(qid,result):
  pred_file="{"
  for i in range(len(qid)):
    score_stage2=result[i]["score"]
    final_score=score_stage2
    pred_file=pred_file+"\""+qid[i]+"\": "
    if final_score>0.36:
      pred_file=pred_file+"\""+result[i]["answer"].replace('"',"\\\"").replace('\n',"")+"\""
    else:
      pred_file=pred_file+"\"\""
    if i!=len(qid)-1:
      pred_file+=", "

  pred_file+="}"
  text_file = open("pred_with2stage.json", "w")
  n = text_file.write(pred_file)
  text_file.close()

def run_all(model_name):
  if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = torch.device('cuda:0')
  else:
      raise Exception("WARNING: Could not find GPU! Using CPU only. \
  To enable GPU, please to go Edit > Notebook Settings > Hardware \
  Accelerator and select GPU.")
  score_stage1=do_stage1(device)
  print("Successfully do stage 1")
  result,qid = get_dev_QA(model_name)
  combine_2_stages(qid,result,score_stage1)
  print("Successfully generate prediction file  with reduced range")

def run_all_whole(model_name):
  if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = torch.device('cuda:0')
  else:
      raise Exception("WARNING: Could not find GPU! Using CPU only. \
  To enable GPU, please to go Edit > Notebook Settings > Hardware \
  Accelerator and select GPU.")
  score_stage1=do_stage1(device)
  print("Successfully do stage 1")
  result,qid = get_dev_QA_whole(model_name)
  combine_2_stages(qid,result,score_stage1)
  print("Successfully generate prediction file without reduced range")

def main(params):
    # print(params.reduced_range)
    if params.reduced_range:
      run_all(params.model_name)
    else:
      run_all_whole(params.model_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2stageQA")
    parser.add_argument("--model_name", type=str)
    parser.add_argument('--reduced_range', default=False, action='store_true')
    parser.add_argument('--whole_range', dest='reduced_range', action='store_false')

    main(parser.parse_args())
