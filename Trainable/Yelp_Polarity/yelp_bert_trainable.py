#Importing all the necessary libraries
from pycm import *
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import pickle
import sys
from glob import glob  
import math
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torchvision.transforms as visionTransforms
import PIL.Image as Image
from torchvision.transforms import ToTensor,ToPILImage
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dfTrainOriginal=pd.read_csv("/root/yelp_review_polarity_csv/train.csv",header=None)
dfTest=pd.read_csv("/root/yelp_review_polarity_csv/train.csv",header=None)

labelEncoder=preprocessing.LabelEncoder()
encodedLabelListTrain=(labelEncoder.fit_transform(dfTrainOriginal[0]))
dfTrainOriginal[0]=encodedLabelListTrain

labelEncoder=preprocessing.LabelEncoder()
encodedLabelListTest=(labelEncoder.fit_transform(dfTest[0]))
dfTest[0]=encodedLabelListTest

dfTrain,dfVal=np.split(dfTrainOriginal.sample(frac=1, random_state=42), [int(.9 * len(dfTrainOriginal))])
dfTrain=dfTrain.reset_index(drop=True)
dfTest=dfTest.reset_index(drop=True)
dfVal=dfVal.reset_index(drop=True)

dfTrain.columns=['Label','Text']
dfVal.columns=['Label','Text']
dfTest.columns=['Label','Text']

from torch.utils.data import Dataset, DataLoader
class DocClassificationDataset(Dataset):

  def __init__(self,dataframe,bertTokenizer,maxLength,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.maxLength=maxLength
    self.pad=self.bertTokenizer('[PAD]',padding='max_length', truncation='longest_first', max_length=64, return_tensors='pt').to(device)

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.txt= str(self.data.iloc[idx,1])
    self.label=self.data.iloc[idx,0]

    sentList=sent_tokenize(self.txt)

    if (len(sentList)<32):
      self.encodedTxt=self.bertTokenizer(sentList,padding='max_length', truncation='longest_first', max_length=64, return_tensors='pt').to(device)
      self.encodedTxt['input_ids']=torch.cat([self.encodedTxt['input_ids'],torch.cat([self.pad['input_ids']]*(32-len(sentList)))]).to(device)
      self.encodedTxt['token_type_ids'] = torch.cat([ self.encodedTxt['token_type_ids'],torch.cat( [self.pad['token_type_ids']] * (32 - len(sentList)))]).to(device)
      self.encodedTxt['attention_mask'] = torch.cat([self.encodedTxt['attention_mask'],torch.zeros(32-len(sentList),64,dtype=torch.int64).to(device)],dim=0).to(device)
      attentionMask=torch.cat((torch.ones(len(sentList),dtype=torch.uint8),torch.zeros(32-len(sentList),dtype=torch.uint8)),dim=0).to(device)


    else:
      self.encodedTxt=self.bertTokenizer(sentList[0:32],padding='max_length', truncation='longest_first', max_length=64, return_tensors='pt').to(device)
      attentionMask = torch.ones(32,dtype=torch.uint8).to(device)

    return self.encodedTxt,self.label,attentionMask

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
docClassificationTrainDataset=DocClassificationDataset(dataframe=dfTrain,bertTokenizer=tokenizer,maxLength=64,device=device)
docClassificationTestDataset=DocClassificationDataset(dataframe=dfTest,bertTokenizer=tokenizer,maxLength=64,device=device)
docClassificationValDataset=DocClassificationDataset(dataframe=dfVal,bertTokenizer=tokenizer,maxLength=64,device=device)

trainLoader=torch.utils.data.DataLoader(docClassificationTrainDataset,batch_size=8,shuffle=True)
testLoader=torch.utils.data.DataLoader(docClassificationTestDataset,batch_size=8,shuffle=True)
valLoader=torch.utils.data.DataLoader(docClassificationValDataset,batch_size=8,shuffle=True)

class BERTOnly(nn.Module):
  def __init__(self,preTrainedBert,embeddingDimension=768,numClasses=2):
    super(BERTOnly,self).__init__()

    self.embDim=embeddingDimension
    self.numClasses=numClasses

    self.dropoutLayer=nn.Dropout(p=0.5)
    self.bert=self.freezeBert(preTrainedBert)

  def meanPooling(self,input_ids,token_type_ids,attention_mask):
    bertOutput=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
    
    #bertOutput=self.bert(input_ids=input['input_ids'].squeeze(dim=1),attention_mask=input['attention_mask'].squeeze(dim=1),token_type_ids=input['token_type_ids'].squeeze(dim=1)).last_hidden_state
    token_embeddings = bertOutput #First element of model_output contains all token embeddings
    #attention_mask=input['attention_mask'].squeeze(dim=0)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

  def forward(self,input_ids,token_type_ids,attention_mask):
    output=self.meanPooling(input_ids,token_type_ids,attention_mask)
    #classificationOutput=self.fc1(self.dropoutLayer(output))
    #classificationOutput=classificationOutput.reshape((classificationOutput.size(0)))

    return output

  def freezeBert(self,model):
    count=0
    for name,params in model.named_parameters():
      count=count+1
      if count<133:
        params.requires_grad=False
    return model

from torch.autograd import Variable

class PositionalEncoder(nn.Module):
  def __init__(self, d_model, max_seq_len = 64):
    super().__init__()
    self.d_model = d_model

    # create constant 'pe' matrix with values dependant on 
    # pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = \
            math.sin(pos / (10000 ** ((2 * i)/d_model)))
            pe[pos, i + 1] = \
            math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
            
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)


  def forward(self, x):
    # make embeddings relatively larger
    x = x * math.sqrt(self.d_model)
    #add constant to embedding
    seq_len = x.size(1)
    x = x + Variable(self.pe[:,:seq_len], \
    requires_grad=False).to(device)
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self, heads, d_model, dropout = 0.5):
    super().__init__()
    
    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads
    
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

  def forward(self, q, k, v, mask=None):
    bs = q.size(0)
    
    # perform linear operation and split into h heads
    
    k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
    q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
    v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
    
    # transpose to get dimensions bs * h * sl * d_model
    
    k = k.transpose(1,2)
    q = q.transpose(1,2)
    v = v.transpose(1,2)# calculate attention using function we will define next
    scores = attention(q, k, v, self.d_k, mask, self.dropout)
    
    # concatenate heads and put through final linear layer
    concat = scores.transpose(1,2).contiguous()\
    .view(bs, -1, self.d_model)
    
    output = self.out(concat)

    return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    
  scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

  #print("hi")
  #print(mask.shape)
  #print(scores.shape)
  if mask is not None:
      mask = mask.unsqueeze(1)
      #print(mask.shape)
      scores = scores.masked_fill(mask == 0, -1e9)
      #print(scores.shape)

  scores = F.softmax(scores, dim=-1)
  #print(scores.shape)

  if dropout is not None:
      scores = dropout(scores)
      
  output = torch.matmul(scores, v)
  return output

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout = 0.5):
    super().__init__() 

    # We set d_ff as a default to 2048
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
  
  def forward(self, x):
    x = self.dropout(F.relu(self.linear_1(x)))
    x = self.linear_2(x)
    return x

class Norm(nn.Module):
  def __init__(self, d_model, eps = 1e-6):
    super().__init__()

    self.size = d_model
    
    # create two learnable parameters to calibrate normalisation
    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    
    self.eps = eps
  
  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
    / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm

class EncoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout = 0.5):
    super().__init__()
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.attn = MultiHeadAttention(heads, d_model)
    self.ff = FeedForward(d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
      
  def forward(self, x, mask):
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
    x2 = self.norm_2(x)
    x = x + self.dropout_2(self.ff(x2))
    return x

import copy
def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
  def __init__(self, d_model, N, heads):
    super().__init__()
    self.N = N
    self.pe = PositionalEncoder(d_model)
    self.layers = get_clones(EncoderLayer(d_model, heads), N)
    self.norm = Norm(d_model)

  def forward(self, src, mask):
    x = self.pe(src)
    for i in range(self.N):
        x = self.layers[i](x, mask)
    return self.norm(x)

class Transformer(nn.Module):
  def __init__(self,d_model=768,N=4,heads=8):
    super().__init__()
    self.encoder=Encoder(d_model=768,N=4,heads=8)
    self.fc=nn.Linear(768,2)


  def forward(self,inputEmb,attn_mask):
    output=self.encoder(inputEmb,attn_mask)
    output=self.mean_pooling(output,attn_mask)
    output=self.fc(output)
    return output


  def mean_pooling(self,model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    #print(attention_mask.shape)
    attention_mask=attention_mask.squeeze(dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class HierNetwork(nn.Module):
  def __init__(self,model):
    super(HierNetwork,self).__init__()
    self.wordBert=BERTOnly(preTrainedBert=model)
    self.sentTransformer=Transformer()
    for p in self.sentTransformer.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self,input,attentionMask):
    count=0
    for temp1,temp2,temp3 in list(zip(input['input_ids'].permute(1,0,2),input['token_type_ids'].permute(1,0,2),input['attention_mask'].permute(1,0,2))):
      count=count+1
      wordLevelOutput=self.wordBert(temp1,temp2,temp3)
      if (count==1):
        finalWordLevelOutput=wordLevelOutput.unsqueeze(dim=0)
      else:
        finalWordLevelOutput=torch.cat((finalWordLevelOutput,wordLevelOutput.unsqueeze(dim=0)))
    finalWordLevelOutput=finalWordLevelOutput.permute(1,0,2)
    sentTransformerOutput=self.sentTransformer(finalWordLevelOutput,attentionMask)

    return sentTransformerOutput

model = BertModel.from_pretrained('bert-base-uncased')
finalNetwork=HierNetwork(model)
finalNetwork.to(device)
softmaxLoss = nn.CrossEntropyLoss()
optimizer = optim.Adam(finalNetwork.parameters(), lr=0.00001) 

def Average(lst): 
    return sum(lst) / len(lst) 

def train_model(model,epochs):

  trainBatchCount=0
  testBatchCount=0

  avgTrainAcc=[]
  avgValidAcc=[]
  trainAcc=[]
  validAcc=[]
  trainLosses=[]
  validLosses=[]
  avgTrainLoss=[]
  avgValidLoss=[]


  for i in range(epochs):

    print("Epoch:",i)

    model.train()
    print("Training.....")
    for batch_idx,(input,targets,attn_masks) in enumerate(trainLoader):

      trainBatchCount=trainBatchCount+1

      targets=targets.to(device)
      
      optimizer.zero_grad()

      scores=model(input,attn_masks)
       
      loss=softmaxLoss(scores,targets)

      loss.backward()

      optimizer.step()

      trainLosses.append(float(loss))
      
      correct=0
      total=0
      total=len(targets)

      predictions=torch.argmax(scores,dim=1)
      correct = (predictions==targets).sum()
      acc=float((correct/float(total))*100)

      trainAcc.append(acc)

      if ((trainBatchCount%200)==0):

        print("Targets:-",targets)
        print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(float(loss), acc))
        print('Batch:',batch_idx)
        print('Epoch:',i)
    model.eval()
    print("Validating.....")
    for batch_idx,(input,targets,attn_masks) in enumerate(valLoader):
      
      testBatchCount=testBatchCount+1

      targets=targets.to(device)

      scores=model(input,attn_masks)      

      loss=softmaxLoss(scores,targets)

      validLosses.append(float(loss))

      testCorrect=0
      testTotal=0

      _,predictions=scores.max(1)

      testCorrect = (predictions==targets).sum()
      testTotal=predictions.size(0)

      testAcc=float((testCorrect/float(testTotal))*100)

      validAcc.append(testAcc)

      if ((testBatchCount%200)==0):
        print('Loss: {}  Accuracy: {} %'.format(float(loss), testAcc))
    

    trainLoss=Average(trainLosses)
    validLoss=Average(validLosses)
    avgTrainLoss.append(trainLoss)
    avgValidLoss.append(validLoss)
    tempTrainAcc=Average(trainAcc)
    tempTestAcc=Average(validAcc)
    avgTrainAcc.append(tempTrainAcc)
    avgValidAcc.append(tempTestAcc)

    print("Epoch Number:-",i,"  ","Training Loss:-"," ",trainLoss,"Validation Loss:-"," ",validLoss,"Training Acc:-"," ",tempTrainAcc,"Validation Acc:-"," ",tempTestAcc)

    trainAcc=[]
    ValidAcc=[]
    trainLosses=[]
    validLosses=[]

  return model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc

finalNetwork,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc = train_model(finalNetwork,3)

loss_train = avgTrainLoss
loss_val = avgValidLoss
epochs = range(1,len(avgTrainLoss)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('bert_trainable_docClassification_YelpPolarity_loss.png', bbox_inches='tight')
plt.close()
#plt.show()

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,len(avgTrainAcc)+1)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('bert_trainable_docClassification_YelpPolarity_acc.png', bbox_inches='tight')
plt.close()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for input,targets,attn_masks in loader:

      targets=targets.to(device=device)

      scores=model(input,attn_masks)
      predictions=torch.argmax(scores,dim=1)

      targets=targets.tolist()
      predictions=predictions.tolist()

      completeTargets.append(targets)
      completePreds.append(predictions)

    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm

cm=checkClassificationMetrics(testLoader,finalNetwork)

f=open("bert_trainable_docClassification_YelpPolarity_Results.txt","a")
f.write(str(cm))
f.close()

