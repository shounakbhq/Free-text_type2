import streamlit as st
import requests
import json
import os
import pandas as pd
from sklearn.metrics import classification_report
qmap={}
qmap[1]=tuple([2])
qmap[2]=tuple([3])
with open('../scorer/questions/qlist.json', 'r') as f:
      qlist = json.load(f)
alist=tuple([a for a in qlist])
# st.session_state['Trained']=False 
# st.write(st.session_state)
# bt_save=False
# if('Trained' in st.session_state):
#     if(st.session_state['Trained']):
#         bt_save=st.sidebar.button('save?')
bt_save=st.sidebar.button('save?')
if('Trained' not in st.session_state):
    bt_save=False
elif('Trained' in st.session_state):
    if(not st.session_state['Trained']):
        bt_save=False

# if 'key' not in st.session_state:
#     # st.session_state['key'] = 'value'
#     st.session_state['key'] = -1
#     st.session_state['role']=-1

train = st.sidebar.checkbox('Train')
validation = st.sidebar.checkbox('Validate')
if((not train) and (not validation)):
    st.session_state['Trained']=False 
    aid = st.selectbox(
        'Please setect the Assessment id',
        alist)
    qid = st.selectbox(
        'Please setect the question id',
    [q for q in qlist[aid]])
    ncol = st.sidebar.number_input("Number of response factors", 0, 50, 1)
    cols = st.columns(ncol)
    data={}
    data['aid']=aid
    data['qid']=qid
    data['answers']=[]
    for i, x in enumerate(cols):
        # x.selectbox(f"Input # {i}",[1,2,3], key=i)
        sent=st.text_input('Please input a text the candidate might give as answer',key=i)
        data['answers'].append(sent)
    bt=st.button('Score')
    addr = 'http://127.0.0.1:8001/scorer/score/'
    if(bt):
        response=requests.post(addr,json=data)
        response=response.content.decode('utf-8')
        print(response)
        response=json.loads(response)
        st.write(response['classes'])
elif(train):
    addr = 'http://127.0.0.1:8001/scorer/train/'
    aid = st.text_input('Assessment id')
    qid = st.text_input('Question id')
    train_excel=st.file_uploader("Choose a training excel file")
    # database=st.file_uploader("Choose a training file")
    # validate=st.file_uploader("Choose a validation file")
    # neg_train=st.file_uploader("Choose a negative file")
    # neg_test=st.file_uploader("Choose a negative validation file")
    # description=st.file_uploader("Choose a description file")
    
    
    data={}
    data['aid']=aid
    data['qid']=qid
    bt=st.button('Train')
    if(bt):
        path='../scorer/training/'+str(aid)
        isExist = os.path.exists(path)
        if not isExist:

           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
        
        path='../scorer/descriptions/'+str(aid)
        isExist = os.path.exists(path)
        if not isExist:

           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
        
        if(train_excel):
            train_excel = pd.read_excel(train_excel,sheet_name=None)
            database=train_excel['database']
            validate=train_excel['validate']
            neg_train=train_excel['neg_train']
            neg_test=train_excel['neg_test']
            description=train_excel['description']
            database_json=database.to_json()
            validate_json=validate.to_json()
            neg_train_json=neg_train.to_json()
            neg_test_json=neg_test.to_json()
            description_json=description.to_json()
            
#         if(database):
#             database = pd.read_excel(database)
#             database_json=database.to_json()
#             # database.to_excel('../scorer/training/'+str(aid)+'/'+qid+'.xlsx')
#         if(validate):
#             validate = pd.read_excel(validate)
#             validate_json=validate.to_json()
#             # validate.to_excel('../scorer/training/'+str(aid)+'/'+qid+'_val.xlsx')

#         if(neg_train):
#             neg_train = pd.read_excel(neg_train)
#             neg_train_json=neg_train.to_json()
#             # neg_train.to_excel('../scorer/training/'+str(aid)+'/'+qid+'_neg.xlsx')

#         if(neg_test):
#             neg_test = pd.read_excel(neg_test)
#             neg_test_json=neg_test.to_json()
#             # neg_test.to_excel('../scorer/training/'+str(aid)+'/'+qid+'_val_neg.xlsx')
        
#         if(description):
#             description = pd.read_excel(description)
#             description_json=description.to_json()
#             # description.to_excel('../scorer/descriptions/'+str(aid)+'/'+str(qid)+'.xlsx')
#         print("Database",database_json)
        
        data['database']=database_json
        data['validate']=validate_json
        data['neg_train']=neg_train_json
        data['neg_test']=neg_test_json
        data['description']=description_json
        
        response=requests.post(addr,json=data)
        response=response.content.decode('utf-8')
        
        print(response)
        response=json.loads(response)
        if(response['status']==200):
            st.write("Trained !!!")
        addr = 'http://127.0.0.1:8001/scorer/validate/'
        data['temp']=True
        response=requests.post(addr,json=data)
        response=response.content.decode('utf-8')
        print(response)
        response=json.loads(response)
        y_test_new=response['y_test']
        prediction_new=response['prediction']
        labels_new=response['labels']
        st.session_state['y_test']=y_test_new
        st.session_state['prediction']=prediction_new
        st.session_state['labels']=labels_new
        
        res=classification_report(y_test_new, prediction_new, labels=labels_new,output_dict=True)
        res=pd.DataFrame(res).transpose()
        st.write("Trained results")
        st.dataframe(res)
        with open('../scorer/questions/qlist.json', 'r') as f:
          qlist = json.load(f)
        if(aid in qlist):
            if(qid in qlist[aid]):
                if('y_test' in qlist[aid][qid]):
                    st.write("Last results")
                    y_test=qlist[aid][qid]['y_test']
                    prediction=qlist[aid][qid]['prediction']
                    labels=qlist[aid][qid]['labels']
                    res=classification_report(y_test, prediction, labels=labels,output_dict=True)
                    res=pd.DataFrame(res).transpose()
                    st.dataframe(res)
        st.session_state['Trained']=True  
print(st.session_state['Trained']) 
if('Trained' in st.session_state):
    if(bt_save and st.session_state['Trained']):
        print("SAVING!!!")
        data={}
        data['aid']=aid
        data['qid']=qid
        data['y_test']=st.session_state['y_test']
        data['prediction']=st.session_state['prediction']
        data['labels']=st.session_state['labels']
        addr = 'http://127.0.0.1:8001/scorer/save/'
        response=requests.post(addr,json=data)
        response=response.content.decode('utf-8')
        response=json.loads(response)
        if(response['status']==200):
            st.write("Saved !!!")
            
            
        
        
if(validation and not train):
    st.session_state['Trained']=False 
    addr = 'http://127.0.0.1:8001/scorer/validate/'
    aid = st.selectbox(
        'Please setect the Assessment id',
        alist)
    qid = st.selectbox(
        'Please setect the question id',
    [q for q in qlist[aid]])
    test_excel=st.file_uploader("Choose a validation excel file")
    # validate=st.file_uploader("Choose a validation file")
    # neg_test=st.file_uploader("Choose a negative validation file")
    data={}
    data['aid']=aid
    data['qid']=qid
    data['temp']=False
    bt=st.button('Validate')
    if(bt):
        path='../scorer/training/'+str(aid)
        isExist = os.path.exists(path)
        if not isExist:

           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
        if(test_excel):
            test_excel = pd.read_excel(test_excel,sheet_name=None)
            validate=test_excel['validate']
            neg_test=test_excel['neg_test']
            validate_json=validate.to_json()
            neg_test_json=neg_test.to_json()
        # if(validate):
        #     validate = pd.read_excel(validate)
        #     validate_json=validate.to_json()
        #     # validate.to_excel('../scorer/training/'+str(aid)+'/'+str(qid)+'_val.xlsx')
        # if(neg_test):
        #     neg_test = pd.read_excel(neg_test)
        #     neg_test_json=neg_test.to_json()
        #     # neg_test.to_excel('../scorer/training/'+str(aid)+'/'+str(qid)+'_val_neg.xlsx')
        data['validate']=validate_json
        data['neg_test']=neg_test_json
        
        response=requests.post(addr,json=data)
        response=response.content.decode('utf-8')
        print(response)
        response=json.loads(response)
        y_test=response['y_test']
        prediction=response['prediction']
        labels=response['labels']
        print(len(y_test),len(prediction))
        res=classification_report(y_test, prediction, labels=labels,output_dict=True)
        res=pd.DataFrame(res).transpose()
        st.dataframe(res)