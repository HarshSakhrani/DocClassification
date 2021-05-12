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
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _find_match
from sklearn.model_selection import train_test_split
import os
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0'

MD5 = 'f7ddfafed1033f68ec72b9267863af6c'

root="/root/DocClassification/Yelp/"

split="train"

NUM_LINES = {
    'train': 650000,
    'test': 50000,
}

_PATH = 'yelp_review_full_csv.tar.gz'

dataset_tar = download_from_url(URL, root=root,
                                path=os.path.join(root, _PATH),
                                hash_value=MD5, hash_type='md5')
extracted_files = extract_archive(dataset_tar)

dfTrainOriginal=pd.read_csv("/root/DocClassification/Yelp/yelp_review_full_csv/train.csv",header=None)
dfTest=pd.read_csv("/root/DocClassification/Yelp/yelp_review_full_csv/test.csv",header=None)

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

  def __init__(self,dataframe,preTrainedSentBert,device):
    self.token = ['[PAD]']
    self.data=dataframe
    self.model=preTrainedSentBert
    self.pad = self.model.encode(self.token, convert_to_tensor=True)

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.txt= str(self.data.iloc[idx,0])
    self.label=self.data.iloc[idx,1]

    sentList=sent_tokenize(self.txt)

    if (len(sentList)<64):
      embeddings=self.model.encode(sentList, convert_to_tensor=True)
      paddedEmbeddings=torch.cat([embeddings, torch.cat([self.pad]* (64 - len(sentList)))])
      attentionMask=torch.cat((torch.ones(len(sentList),dtype=torch.uint8),torch.zeros(64-len(sentList),dtype=torch.uint8)),dim=0).to(device)


    else:
      paddedEmbeddings=self.model.encode(sentList[0:64],convert_to_tensor=True)
      attentionMask = torch.ones(64,dtype=torch.uint8).to(device)

    return paddedEmbeddings,attentionMask,self.label

from sentence_transformers import SentenceTransformer, util
preTrainedSentBert = SentenceTransformer('bert-base-nli-stsb-mean-tokens', device=device)
for name,params in preTrainedSentBert.named_parameters():
  params.requires_grad=False

docClassificationTrainDataset=DocClassificationDataset(dataframe=dfTrain,preTrainedSentBert=preTrainedSentBert,device=device)
docClassificationTestDataset=DocClassificationDataset(dataframe=dfTest,preTrainedSentBert=preTrainedSentBert,device=device)
docClassificationValDataset=DocClassificationDataset(dataframe=dfVal,preTrainedSentBert=preTrainedSentBert,device=device)

trainLoader=torch.utils.data.DataLoader(docClassificationTrainDataset,batch_size=8,shuffle=True)
testLoader=torch.utils.data.DataLoader(docClassificationTestDataset,batch_size=8,shuffle=True)
valLoader=torch.utils.data.DataLoader(docClassificationValDataset,batch_size=8,shuffle=True)

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
    self.fc=nn.Linear(768,5)


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


model=Transformer()

for p in model.parameters():
  if p.dim() > 1:
    nn.init.xavier_uniform_(p)

model.to(device)

softmaxLoss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

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
    for batch_idx,(inputEmb,attn_masks,targets) in enumerate(trainLoader):

      trainBatchCount=trainBatchCount+1

      targets=targets.to(device)
      
      optimizer.zero_grad()

      scores=model(inputEmb,attn_masks)
       
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

    model.eval()
    print("Validating.....")
    for batch_idx,(inputEmb,attn_masks,targets) in enumerate(valLoader):

      testBatchCount=testBatchCount+1

      targets=targets.to(device)

      scores=model(inputEmb,attn_masks)      

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

model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc = train_model(model,10)

loss_train = avgTrainLoss
loss_val = avgValidLoss
epochs = range(1,len(avgTrainLoss)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('YelpLoss_4.png', bbox_inches='tight')
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
plt.savefig('YelpAcc_4.png', bbox_inches='tight')
plt.close()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for inputEmb,attn_masks,targets in loader:

      targets=targets.to(device=device)

      scores=model(inputEmb,attn_masks)
      predictions=torch.argmax(scores,dim=1)

      targets=targets.tolist()
      predictions=predictions.tolist()

      completeTargets.append(targets)
      completePreds.append(predictions)

    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm

cm=checkClassificationMetrics(testLoader,model)

f=open("Yelp_4_Results.txt","a")
f.write(str(cm))
f.close()

