from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from keybert import KeyBERT
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re
import math
from operator import itemgetter
import fitz
import json
import nltk 
import os
import numpy as np
from nltk import pos_tag, word_tokenize, RegexpParser 
from transformers import create_optimizer
from transformers import TFAutoModelForSequenceClassification,RobertaTokenizer,RobertaModel
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer,AutoConfig,AutoModel
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
import random
from datasets import load_metric
from transformers import AdamW,get_scheduler
from collections import Counter
from tqdm.auto import tqdm
from collections import Counter,OrderedDict
import math
import os
import openai
import requests
from sentence_transformers import SentenceTransformer
import pickle
from scorer.src.models.wrapper import DIETClassifierWrapper
config_file = "scorer/config.yml"
wrapper = DIETClassifierWrapper(config=config_file)
openai.api_key = os.getenv("OPENAI_API_KEY")

metric = load_metric("accuracy")
metric1 = load_metric("precision")
metric2 = load_metric("recall")
metric3= load_metric("f1")
rasa_intent=["Reason for high attrition rate","Given the sales team are the most extroverted people, sales team is the only department that is facing high attrition rate.","It is a long term issue since most of the company is in favour of remote working.","Sales reps have lost healthy competition lowering their motivation to work. Hence the company is looking for solution to turn around this issue","The immediate focus of HR team is to improve retention rate, and the sales team is focusing to maintain meeting its sales target.","Not belong to any class"]
with open('scorer/models_entailment/rasa-test/svm_rel_irl.sav', 'rb') as pickle_file:
        model_rel_irl = pickle.load(pickle_file)
model_base=SentenceTransformer('paraphrase-mpnet-base-v2')


class PartModel(nn.Module):
  def __init__(self,num_labels): 
    super(PartModel,self).__init__() 
    self.num_labels = num_labels 
    self.l1=nn.Linear(768, 768)
    self.l2=nn.Linear(768, 768)
    self.softmax = nn.LogSoftmax(dim = 1) 
    self.dropout = nn.Dropout(0.2) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, encoded):
    out=self.l1(encoded)
    out=self.dropout(out)
    out2=self.l2(out)
    out2=self.dropout(out2)
    # print(out.shape)
    logits = self.classifier(out2[:,:].view(-1,768)) # calculate losses
    x = self.softmax(logits)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
tokenizer.model_max_len=512

checkpoint='sentence-transformers/paraphrase-mpnet-base-v2'
model_ckpt=AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)).to(device)

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def preprocess_function_entailment(examples):
    return tokenizer(examples["text1"],examples["text2"], truncation=True)

def kt_model(sents,desc,model_part):
    X_test_1=[]
    X_test_2=[]
    start=0
    ch_ind=[]
    # print("desc",desc)
    for row in sents:
        count=0
        for t in range(len(desc)):
             for des in desc[t]:
                  count+=1
                  X_test_1.append(row)
                  # X_train_2.append(desc[t])
                  X_test_2.append(des)
                  # y_test_ret.append(0.0)
                #  X_test_1.append(row)
                #  X_test_2.append(desc[t])
                #  y_test_ret.append(0.0)

        ch_ind.append([start,start+count])
        start+=count
    
    # print(X_test_1,X_test_2)
    
    d = {'test':Dataset.from_dict({'text1':X_test_1,'text2':X_test_2})}
    
    d=DatasetDict(d)
    
    tokenized_imdb = d.map(preprocess_function_entailment, batched=True)
    tokenized_imdb.set_format("torch",columns=["input_ids", "attention_mask"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataloader = DataLoader(
        tokenized_imdb["test"], batch_size=50, collate_fn=data_collator
    )
    model_part.eval()
    conf=[]
    p=[]
    l=[]
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            pred=model_ckpt(**batch)
            a = mean_pool(pred[0], batch['attention_mask'])
            outputs = model_part(a)
            _, predictions = torch.max(outputs.data, 1)
            # print("_",_)
            # print("pre",predictions)
            # print(outputs.data)
        # l=[]
        logits = outputs.data
        m = nn.Softmax(dim=1)
        input = logits
        out = m(input)
        conf.append(out)
        # print(out[0])
        predictions = torch.argmax(logits, dim=-1)
        p.append(predictions)
        # l.append(batch["labels"])
        # metric.add_batch(predictions=predictions, references=batch["labels"])
        # metric1.add_batch(predictions=predictions, references=batch["labels"])
        # metric2.add_batch(predictions=predictions, references=batch["labels"])
        # metric3.add_batch(predictions=predictions, references=batch["labels"])

    p = [x.cpu().numpy().item() for xs in p for x in xs]
    # l = [x.cpu().numpy().item() for xs in l for x in xs]
    flat_list = [list(x.cpu().numpy()) for xs in conf for x in xs]
    gap=[]
    for des in desc:
      gap.append(len(des))
    
    X_test=[]
    prediction=[]
    # for i in range(int(len(X_test_1)/len(desc))):
    for i in range(len(ch_ind)):
      # chunk=conf[0][len(desc)*i:len(desc)*i+len(desc)]
      # chunk=flat_list[len(desc)*i:len(desc)*i+len(desc)]
      chunk=flat_list[ch_ind[i][0]:ch_ind[i][1]]
      start=0
      chunk_new=[]
      # print("frist chunk",chunk)
      for _,g in enumerate(gap):
        chunk_new.append(chunk[start:start+gap[_]])
        # print("to_app",start,gap[_],chunk[start:start+gap[_]])
        start+=gap[_]
      ma=0
      th=0.5
      X_test.append(X_test_1[ch_ind[i][0]])
      # print("chunk_new",chunk_new)
      for j,chunk in enumerate(chunk_new):
        cur=0
        # print("chunk",chunk)
        for logits in (chunk):
          # print("compare",X_test_1[ch_ind[i][0]],desc[int(j)],logits[1])
          cur=max(cur,logits[1])
          # print("cur",cur)
        # cur/=len(chunk)
        if(cur>ma):
          pred_cur=j
        ma=max(cur,ma)
      # print(ma)
      if(ma>th):
        prediction.append([pred_cur,ma])
      else:
        prediction.append([-1,ma])
    return prediction

@csrf_exempt
def score(request):
    reqs = request.body.decode('utf-8')
    # print(reqs)
    print(reqs)
    # reqjson=reqs
    reqjson = json.loads(reqs)
    # st1=time.time()
    qid=reqjson['qid']
    aid=reqjson['aid']
    answers=reqjson['answers']
    if(False):
        # for answer in answers:
        pass
            
    else:
        gpt_res=[]
        for answer in answers:
          # response = openai.Completion.create(
          # model="text-davinci-003",
          # prompt="decide whether the sentence talks about monthly user fees, number of customers/users, customer life time or others:\nsentence: "+answer+'\n'
          # ,
          # temperature=0,
          # max_tokens=20,
          # top_p=1,
          # frequency_penalty=0.0,
          # presence_penalty=0.0
          # )
          # print("response",response["choices"][0]["text"])
          if(len(answer.strip().split())<2):
            gpt_res.append(1)
          # elif(response["choices"][0]["text"].strip().lower().find("others")!=-1):
          #   gpt_res.append(1)
          else:
            gpt_res.append(0)
        # for ans in answers:
        #     classes=predict(ans,aid,qid)
        
        description_file=pd.read_excel('scorer/descriptions/'+str(aid)+'/'+str(qid)+'.xlsx')
        if('Unnamed: 0' in description_file):
            description_file=description_file.drop(['Unnamed: 0'], axis=1)
        description=[]
        for col in description_file:
            temp=[]
            for row in description_file[col]:
                if(type(row)==str):
                    row=row.replace(u'\xa0', u' ')
                    temp.append(row)
            description.append(temp)
        
        
        result={}
        if(aid=='rasa-test'):
            prediction=[]
            lab=[i for i in range(len(description))]
            lab.append(-1)
            description.append(["Not belong to any class"])
            for x in answers:
                pred_flag=0
                res=requests.post("http://127.0.0.1:5005/model/parse",json={"text":x})
                # print(res.content)
                res = res.content.decode('utf-8')
                res = json.loads(res)
                if(res["intent"]["confidence"]<0.7):
                    prediction.append(-1)
                    continue
                elif(res["intent"]["confidence"]<0.7 and (res["intent_ranking"][0]["confidence"]-res["intent_ranking"][1]["confidence"])<0.4):
                    prediction.append(-1)
                    continue
                print("RESS!!",res["intent"]["name"])
                for i,clas in enumerate(description):
                    if clas[0]==res["intent"]["name"]:
                        pred_flag=1
                        print(x,"Match!!")
                        if(i==len(description)-1):
                            prediction.append(-1)
                        else:
                            prediction.append(i)
                if(res["intent"]["name"]=="nlu_fallback"):
                    pred_flag=1
                    prediction.append(-1)
                if(pred_flag==0):
                    print("NOOOOOTTTTT!",res["intent"]["name"])
            result['classes']=prediction
        else:
            # description_file=pd.read_excel('scorer/descriptions/'+str(aid)+'/'+str(qid)+'.xlsx')
            # if('Unnamed: 0' in description_file):
            #     description_file=description_file.drop(['Unnamed: 0'], axis=1)
            # description=[]
            # for col in description_file:
            #     temp=[]
            #     for row in description_file[col]:
            #         if(type(row)==str):
            #             row=row.replace(u'\xa0', u' ')
            #             temp.append(row)
            #     description.append(temp)
            entailment_model=torch.load('scorer/models_entailment/'+str(aid)+'/'+str(qid)+'.pt')
            entailment_model=entailment_model.to(device)

            ret=kt_model(answers,description,entailment_model)
            # result={}
            print("ret",ret)
            result['classes']=[a[0] for a in ret]
        for i,x in enumerate(result['classes']):
            if(gpt_res[i]==1):
                result['classes'][i]=-1
    return JsonResponse(result)

@csrf_exempt
def train(request):
    reqs = request.body.decode('utf-8')
    # print(reqs)
    reqjson = json.loads(reqs)
    # st1=time.time()
    qid=reqjson['qid']
    aid=reqjson['aid']
    database=reqjson['database']
    validate=reqjson['validate']
    neg_train=reqjson['neg_train']
    neg_test=reqjson['neg_test']
    description=reqjson['description']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint='sentence-transformers/paraphrase-mpnet-base-v2'
    model=AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)).to(device)
    # database=pd.read_excel('scorer/training/'+str(aid)+'/'+qid+'.xlsx')
    # validate=pd.read_excel('scorer/training/'+str(aid)+'/'+qid+'_val.xlsx')
    # neg_train=pd.read_excel('scorer/training/'+str(aid)+'/'+qid+'_neg.xlsx')
    # neg_test=pd.read_excel('scorer/training/'+str(aid)+'/'+qid+'_val_neg.xlsx')
    # description_file=pd.read_excel('scorer/descriptions/'+str(aid)+'/'+str(qid)+'.xlsx')
    # database=pd.DataFrame.from_dict(database, orient="index")
    print("desc_json",description)
    database = pd.read_json(database, orient ='columns')
    validate = pd.read_json(validate, orient ='columns')
    neg_train = pd.read_json(neg_train, orient ='columns')
    neg_test = pd.read_json(neg_test, orient ='columns')
    description_file = pd.read_json(description, orient ='columns')
    
    
    # validate=pd.DataFrame.from_dict(validate,  orient="index")
    # neg_train=pd.DataFrame.from_dict(neg_train, orient="index")
    # neg_test=pd.DataFrame.from_dict(neg_test, orient="index")
    # description=pd.DataFrame.from_dict(description, orient="index")
    
    
     
    if('Unnamed: 0' in validate):
        validate=validate.drop(['Unnamed: 0'], axis=1)
    if('Unnamed: 0' in neg_train):
        neg_train=neg_train.drop(['Unnamed: 0'], axis=1)
    if('Unnamed: 0' in neg_test):
        neg_test=neg_test.drop(['Unnamed: 0'], axis=1)
    if('Unnamed: 0' in description_file):
        description_file=description_file.drop(['Unnamed: 0'], axis=1)
    desc=[]
    for col in description_file:
        temp=[]
        print("COLLLL",col)
        for row in description_file[col]:
            print("ROOOOWW",row)
            if(type(row)==str):
                row=row.replace(u'\xa0', u' ')
                temp.append(row)
        desc.append(temp)
    description_file.to_excel('scorer/descriptions/'+str(aid)+'/'+str(qid)+'.xlsx')
    # print("description",description_file)
    # print("database",descriatabase)
    # print("validate",validate)
    
    
    X_train_1=[]
    X_train_2=[]
    y_train_ret=[]
    X_test_1=[]
    X_test_2=[]
    y_test_ret=[]
    train_examples = [] 
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    count=0
    database.fillna('',inplace=True)
    validate.fillna('',inplace=True)
    neg_train.fillna('',inplace=True)
    neg_test.fillna('',inplace=True)
    # print("database",database)
    # print("validate",validate)
    # print("database",database)
    # print("database",database)
    
    for i,col in enumerate(neg_train):
      print("COL",col) 
      for _,row in enumerate(neg_train[col]):
        if(row!=''):
          print(description_file.columns)
          for des in desc[i]:
            print("Row",row)
            X_train_1.append(str(row))
            X_train_2.append(des)
            y_train_ret.append(0)
    for i,col in enumerate(database):
        print("COL",col)
        for _,row in enumerate(database[col]):
          if(row!=''):
            for t in range(len(database.columns)):
              if(i!=t):
                 for des in desc[t]:
                  print("Row",row)
                  X_train_1.append(str(row))
                  # X_train_2.append(desc[t])
                  X_train_2.append(des)
                  y_train_ret.append(0.0)
              else:
                for des in desc[i]:
                  print("Row",row)    
                  X_train_1.append(str(row))
                  # X_train_2.append(desc[i])
                  X_train_2.append(des)
                  y_train_ret.append(1.0)
            a=np.array(database[col])
            print(np.arange(len(a))!=_)
            b = a[np.arange(len(a))!=_]
            b=[bx for bx in b if bx!='']
            fin=random.sample(b, min(len(database.columns)-1,len(b)))
            print(len(fin))
            for f in fin:
              if(f!=''):
                count+=1
                X_train_1.append(str(row))
                X_train_2.append(f)
                y_train_ret.append(1)
    start=0
    ch_ind=[]
    for i,col in enumerate(neg_test):
      for _,row in enumerate(neg_test[col]):
        count=0
        if(row!=''):
          y_test.append(-1)
          for t in range(len(validate.columns)):
              if(i!=t):
                 for des in desc[t]:
                  count+=1
                  X_test_1.append(str(row))
                  # X_train_2.append(desc[t])
                  X_test_2.append(des)
                  y_test_ret.append(0.0)
              else:
                for des in desc[i]:
                  count+=1
                  X_test_1.append(str(row))
                  X_test_2.append(des)
                  y_test_ret.append(0.0)
          ch_ind.append([start,start+count])
          start+=count
    for i,col in enumerate(validate):
        for _,row in enumerate(validate[col]):
          count=0
          if(row!=''):
            y_test.append(i)
            for t in range(len(validate.columns)):
              if(i!=t):
                 for des in desc[t]:
                  count+=1
                  X_test_1.append(str(row))
                  X_test_2.append(des)
                  y_test_ret.append(0.0)
              else:
                for des in desc[i]:
                  count+=1
                  X_test_1.append(str(row))
                  X_test_2.append(des)
                  y_test_ret.append(1.0)
            ch_ind.append([start,start+count])
            start+=count
    
    print("X_train_1",X_train_1)
    print("X_train_2",X_train_2)
    print("y_train_ret",y_train_ret)
    print("description",desc)
    
    d = {'train':Dataset.from_dict({'label':y_train_ret,'text1':X_train_1,'text2':X_train_2}),
         'test':Dataset.from_dict({'label':y_test_ret,'text1':X_test_1,'text2':X_test_2})
     }

    d=DatasetDict(d)
    tokenized_imdb = d.map(preprocess_function_entailment, batched=True)
    tokenized_imdb.set_format("torch",columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_imdb["train"], shuffle=True, batch_size=50, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_imdb["test"], batch_size=50, collate_fn=data_collator
    )
    model_part=PartModel(2).to(device)
    # metric = load_metric("accuracy")
    # metric1 = load_metric("precision")
    # metric2 = load_metric("recall")
    # metric3= load_metric("f1")
    optimizer = AdamW(model_part.parameters(), lr=5e-4)
    num_epochs = 100
    batch_size=50
    num_training_steps = num_epochs * (len(X_train_1)/batch_size)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=30,
        num_training_steps=num_training_steps,
    )
    best=0
    best_model=model_part
    ratios=Counter(y_train_ret)
    weights = OrderedDict(sorted(ratios.items()))
    weights= [weights[wt] for wt in weights]
    weights=[1/wt for wt in weights]
    wt_sum=sum(weights)
    weights=[wt/wt_sum for wt in weights]
    progress_bar_train = tqdm(range(math.ceil(num_training_steps)))
    progress_bar_eval = tqdm(range(math.ceil(num_epochs * (len(X_train_1)/batch_size))))
    criterion = nn.NLLLoss(weight=torch.Tensor(weights).to(device),reduction='sum')
    for epoch in range(int(num_epochs)):
      loss_train=0
      loss_test=0
      model_part.train()
      for batch in train_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          # print(batch)
          pred=model(**batch)
          # print(pred[0])
          a = mean_pool(pred[0], batch['attention_mask'])
          outputs = model_part(a)
          _, predictions = torch.max(outputs.data, 1)

          # logits = outputs.logits

          # predictions = torch.argmax(logits, dim=-1)
          metric.add_batch(predictions=predictions, references=batch["labels"])
          metric3.add_batch(predictions=predictions, references=batch["labels"])

          # loss = outputs.loss
          loss = criterion(outputs.to(device), batch['labels'].type(torch.LongTensor).to(device))
          loss_train+=loss
          loss.backward()

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar_train.update(1)
      x=metric.compute()
      y=metric3.compute()
      print(epoch," traint ",x)
      print(epoch," traint_F1 ",y)
      print("train loss", loss_train/len(X_train_1))


      model_part.eval()
      loss_test=0
      for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs=0
        with torch.no_grad():
            pred=model(**batch)
            a = mean_pool(pred[0], batch['attention_mask'])
            outputs = model_part(a)
            _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, batch['labels'].type(torch.LongTensor).to(device))
        loss_test+=loss

        # logits = outputs.logits
        # predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        metric1.add_batch(predictions=predictions, references=batch["labels"])
        metric2.add_batch(predictions=predictions, references=batch["labels"])
        metric3.add_batch(predictions=predictions, references=batch["labels"])


        progress_bar_eval.update(1)
      x=metric.compute()
      metric1.compute()
      metric2.compute()
      print("test loss", loss_test/len(X_test_1))

      # precision = metric1.compute(predictions=predictions, references=batch["labels"],average='weighted')
      # recall = metric2.compute(predictions=predictions, references=batch["labels"],average='micro')
      # f1= metric3.compute(predictions=predictions, references=batch["labels"],average='weighted')
      f1=metric3.compute()

      print(epoch," test ",x)
      print(epoch," test_f1 ",f1['f1'])



      if(f1['f1']>best):
        best_model=model_part
        best=f1['f1']
        # torch.save(model.state_dict(), '/content/drive/MyDrive/HireQuotient/best')
        path='scorer/models_entailment/temp/'+str(aid)
        isExist = os.path.exists(path)
        if not isExist:

           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
        torch.save(model_part, path+'/'+str(qid)+'.pt')

    
#     with open('scorer/questions/qlist.json', 'r') as f:
#       qlist = json.load(f)
#     if(aid in qlist):
#         # qlist[aid].append(qid)
#         if(qid not in  qlist[aid]):
#             qlist[aid][qid]={}
        
#     else:
#         qlist[aid]={}
#         qlist[aid][qid]={}
#     with open('scorer/questions/qlist.json', "w") as outfile:
#         json.dump(qlist, outfile)
    return JsonResponse({'status':200})

@csrf_exempt
def save(request):
    reqs = request.body.decode('utf-8')
    # print(reqs)
    reqjson = json.loads(reqs)
    # st1=time.time()
    qid=reqjson['qid']
    aid=reqjson['aid'] 
    y_test=reqjson['y_test']
    prediction=reqjson['prediction']
    labels=reqjson['labels']
    with open('scorer/questions/qlist.json', 'r') as f:
      qlist = json.load(f)
    if(aid in qlist):
        # qlist[aid].append(qid)
        if(qid not in  qlist[aid]):
            qlist[aid][qid]={}
        
    else:
        qlist[aid]={}
        qlist[aid][qid]={}
    qlist[aid][qid]={}
    qlist[aid][qid]['y_test']=y_test
    qlist[aid][qid]['prediction']=prediction
    qlist[aid][qid]['labels']=labels
    with open('scorer/questions/qlist.json', "w") as outfile:
        json.dump(qlist, outfile)
    path='scorer/models_entailment/'+str(aid)
    isExist = os.path.exists(path)
    if not isExist:

       # Create a new directory because it does not exist
       os.makedirs(path)
       print("The new directory is created!")
    os.replace('scorer/models_entailment/temp/'+str(aid)+'/'+str(qid)+'.pt','scorer/models_entailment/'+str(aid)+'/'+str(qid)+'.pt')
    return JsonResponse({'status':200})
    
@csrf_exempt
def validate(request):
    reqs = request.body.decode('utf-8')
    reqjson = json.loads(reqs)
    qid=reqjson['qid']
    aid=reqjson['aid']
    temp_model=reqjson['temp']
    # database=reqjson['database']
    validate=reqjson['validate']
    # neg_train=reqjson['neg_train']
    neg_test=reqjson['neg_test']
    # description=reqjson['description']
    # validate=pd.read_excel('scorer/training/'+str(aid)+'/'+str(qid)+'_val.xlsx')
    # neg_test=pd.read_excel('scorer/training/'+str(aid)+'/'+str(qid)+'_val_neg.xlsx')
    checkpoint='sentence-transformers/paraphrase-mpnet-base-v2'
    model=AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)).to(device)
    validate = pd.read_json(validate, orient ='columns')
    neg_test = pd.read_json(neg_test, orient ='columns')
    if('Unnamed: 0' in validate):
        validate=validate.drop(['Unnamed: 0'], axis=1)
    if('Unnamed: 0' in neg_test):
            neg_test=neg_test.drop(['Unnamed: 0'], axis=1)
    description_file=pd.read_excel('scorer/descriptions/'+str(aid)+'/'+str(qid)+'.xlsx')
    if('Unnamed: 0' in description_file):
        description_file=description_file.drop(['Unnamed: 0'], axis=1)
    
    desc=[]
    gpt_res=[]
    for col in description_file:
        temp=[]
        for row in description_file[col]:
            if(type(row)==str):
                row=row.replace(u'\xa0', u' ')
                temp.append(row)
        desc.append(temp)
    X_test_1=[]
    X_test_2=[]
    y_test_ret=[]
    X_test=[]
    y_test=[]
    count=0
    validate.fillna('',inplace=True)
    neg_test.fillna('',inplace=True)
    start=0
    ch_ind=[]
    X_all=[]
    for i,col in enumerate(neg_test):
      for _,row in enumerate(neg_test[col]):
        count=0
        if(row!=''):
          X_all.append(row)
          if(len(row.strip().split())<2):
              gpt_res.append(1)
          else:
              gpt_res.append(0)
          y_test.append(-1)
          for t in range(len(validate.columns)):
              if(i!=t):
                 for des in desc[t]:
                  count+=1
                  X_test_1.append(str(row))
                  # X_train_2.append(desc[t])
                  X_test_2.append(des)
                  y_test_ret.append(0.0)
              else:
                for des in desc[i]:
                  count+=1
                  X_test_1.append(str(row))
                  X_test_2.append(des)
                  y_test_ret.append(0.0)
          ch_ind.append([start,start+count])
          start+=count
    for i,col in enumerate(validate):
        for _,row in enumerate(validate[col]):
          count=0
          if(row!=''):
            X_all.append(row)
            if(len(row.strip().split())<2):
                gpt_res.append(1)
            else:
                gpt_res.append(0)
            y_test.append(i)
            for t in range(len(validate.columns)):
              if(i!=t):
                 for des in desc[t]:
                  count+=1
                  X_test_1.append(str(row))
                  X_test_2.append(des)
                  y_test_ret.append(0.0)
              else:
                for des in desc[i]:
                  count+=1
                  X_test_1.append(str(row))
                  X_test_2.append(des)
                  y_test_ret.append(1.0)
            ch_ind.append([start,start+count])
            start+=count
    print(X_all)
    pred_rel_irl=model_rel_irl.predict(model_base.encode(X_all))
    if(aid=='rasa-test' and qid=='2'):
#         #predict
#         rasa_output=wrapper.predict(X_all)
#         prediction=[]
#         lab=[i for i in range(len(desc))]
#         lab.append(-1)
#         desc.append(["Not belong to any class"])
#         for ri,rasa_pred in enumerate(rasa_output):
#             pred_flag=0
#             if(rasa_pred["intent"] in ["nlu_fallback","None",None]):
#                 pred_flag=1
#                 prediction.append(-1)
#                 continue
#             print(type(rasa_pred["intent"]))
#             if(rasa_pred['intent_ranking'][rasa_pred['intent']]<0.7):
#                 prediction.append(-1)
#                 continue
#             # intent_ranking=dict(sorted(rasa_pred['intent_ranking'].items(), key=lambda item: item[1]))
#             # if()
#             for i,clas in enumerate(desc):
#                 if clas[0]==rasa_pred['intent']:
#                     pred_flag=1
#                     print(X_all[ri],"Match!!")
#                     if(i==len(desc)-1):
#                         prediction.append(-1)
#                     else:
#                         prediction.append(i)
            
#             if(pred_flag==0):
#                 print("NOOOOOTTTTT!",rasa_pred["intent"])
            
        
        prediction=[]
        lab=[i for i in range(len(desc))]
        lab.append(-1)
        desc.append(["Not belong to any class"])
        for x in X_all:
            pred_flag=0
            res=requests.post("http://127.0.0.1:5005/model/parse",json={"text":x})
            # print(res.content)
            res = res.content.decode('utf-8')
            res = json.loads(res)
            if(res["intent"]["confidence"]<0.7):
                prediction.append(-1)
                continue
            elif(res["intent"]["confidence"]<0.7 and (res["intent_ranking"][0]["confidence"]-res["intent_ranking"][1]["confidence"])<0.4):
                prediction.append(-1)
                continue
            print("RESS!!",res["intent"]["name"])
            for i,clas in enumerate(desc):
                if clas[0]==res["intent"]["name"]:
                    pred_flag=1
                    print(x,"Match!!")
                    if(i==len(desc)-1):
                        prediction.append(-1)
                    else:
                        prediction.append(i)
            if(res["intent"]["name"]=="nlu_fallback"):
                pred_flag=1
                prediction.append(-1)
            if(pred_flag==0):
                print("NOOOOOTTTTT!",res["intent"]["name"])
    
    else:
        lab=[i for i in range(len(desc))]
        lab.append(-1)
        d = {'test':Dataset.from_dict({'label':y_test_ret,'text1':X_test_1,'text2':X_test_2})
         }

        d=DatasetDict(d)
        tokenized_imdb = d.map(preprocess_function_entailment, batched=True)
        tokenized_imdb.set_format("torch",columns=["input_ids", "attention_mask", "label"])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if(temp_model):
            model_part=torch.load('scorer/models_entailment/temp/'+str(aid)+'/'+str(qid)+'.pt')
        else:
            model_part=torch.load('scorer/models_entailment/'+str(aid)+'/'+str(qid)+'.pt')
        model_part.eval()
        # metric = load_metric("accuracy")
        # metric1 = load_metric("precision")
        # metric2 = load_metric("recall")
        # metric3= load_metric("f1")

        test_dataloader = DataLoader(
            tokenized_imdb["test"], batch_size=50, collate_fn=data_collator
        )
        conf=[]
        p=[]
        l=[]
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                pred=model(**batch)
                a = mean_pool(pred[0], batch['attention_mask'])
                outputs = model_part(a)
                _, predictions = torch.max(outputs.data, 1)
                # print("_",_)
                # print("pre",predictions)
                # print(outputs.data)
            # l=[]
            logits = outputs.data
            m = nn.Softmax(dim=1)
            input = logits
            out = m(input)
            conf.append(out)
            # print(out[0])
            predictions = torch.argmax(logits, dim=-1)
            p.append(predictions)
            l.append(batch["labels"])
            metric.add_batch(predictions=predictions, references=batch["labels"])
            metric1.add_batch(predictions=predictions, references=batch["labels"])
            metric2.add_batch(predictions=predictions, references=batch["labels"])
            metric3.add_batch(predictions=predictions, references=batch["labels"])

        p = [x.cpu().numpy().item() for xs in p for x in xs]
        l = [x.cpu().numpy().item() for xs in l for x in xs]
        flat_list = [list(x.cpu().numpy()) for xs in conf for x in xs]
        print(flat_list)
        print(p)
        print(l)
        print(metric.compute())
        print(metric1.compute(average='micro'))
        print(metric2.compute(average='micro'))
        metric3.compute(average='micro')
        gap=[]
        for des in desc:
          gap.append(len(des))
        X_test=[]
        prediction=[]
        max_array=[]
        # for i in range(int(len(X_test_1)/len(desc))):
        for i in range(len(ch_ind)):
          # chunk=conf[0][len(desc)*i:len(desc)*i+len(desc)]
          # chunk=flat_list[len(desc)*i:len(desc)*i+len(desc)]
          chunk=flat_list[ch_ind[i][0]:ch_ind[i][1]]
          start=0
          chunk_new=[]
          print("frist chunk",chunk)
          for _,g in enumerate(gap):
            chunk_new.append(chunk[start:start+gap[_]])
            print("to_app",start,gap[_],chunk[start:start+gap[_]])
            start+=gap[_]
          ma=0
          th=0.0
          X_test.append(X_test_1[ch_ind[i][0]])
          print("chunk_new",chunk_new)
          for j,chunk in enumerate(chunk_new):
            cur=0
            print("chunk",chunk)
            for logits in (chunk):
              print("compare",X_test_1[ch_ind[i][0]],desc[int(j)],logits[1])
              cur=max(cur,logits[1])
              print("cur",cur)
            # cur/=len(chunk)
            if(cur>ma):
              pred_cur=j
            ma=max(cur,ma)
          print(ma)
          max_array.append(ma)
          if(ma>th):
            prediction.append(pred_cur)
          else:
            prediction.append(-1)
        for i,pred in enumerate(prediction):
            if(gpt_res[i]==1):
                prediction[i]=-1
    
    to_ret={}
    to_ret['y_test']=y_test
    to_ret['prediction']=prediction
    print("GPT_RES",gpt_res)
    print("Desc",len(desc))
    print("FU",[i for i in range(len(desc))].append(-1))
    # lab=[i for i in range(len(desc))]
    # print("QID",qid,"Aid",aid)
    # lab.append(-1)
    # to_ret['labels']=[i for i in range(len(desc))].append(-1)
    to_ret['labels']=lab
    print("labels",to_ret["labels"])
    if(aid=='rasa-test' and qid=='2'):
        for _,pr in enumerate(pred_rel_irl):
            if(pr==0):
                to_ret['prediction'][_]=-1
    return JsonResponse(to_ret)
    
    
    
    

    
        
        