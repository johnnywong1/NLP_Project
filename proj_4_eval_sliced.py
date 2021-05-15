from proj_3_train import ProjModel, ProjData
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset,DataLoader
import logging
from datetime import datetime
import sys

turn = int(sys.argv[1])

def get_neg_dataset(path="/bigtemp/rm5tx/nlp_project/2016-06_all.csv"):
    print("reading neg")
    data = pd.read_csv(path, dtype={'body':str},skiprows=range(1,1000000*turn),nrows=1000000)
    data.rename(columns={"body":"data"}, inplace=True)
    #data["label"] = 0

    return data

def main():
    out_path="/bigtemp/rm5tx/nlp_project/2016-06_all_predicted_"+str(turn+1)+".csv"
    model = ProjModel.load_from_checkpoint(checkpoint_path=os.path.expanduser("~/saved_models/last.ckpt"))
    
    DATA_PATH = os.path.expanduser("~/data_cache/")
    
    logging.basicConfig(filename='log_inference_sliced.log', filemode='w', format='%(message)s', level=logging.INFO)
    mylogger = logging.getLogger('Admin_Client')
    mylogger.info('hello')
    
    data = ProjData(max_len=128)
    data.load(DATA_PATH)
    
    neg_data = get_neg_dataset()
    neg_data.dropna(subset=['data'],inplace=True)
    print(neg_data.shape)
    
    neg_data['data'] = neg_data['data'].map(data.preprocess)
    #print(neg_data['data'].tolist())
    
    #data.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, do_lower_case=True)
    mylogger.info('masking')
    X = neg_data['data']
    X_input_ids = data.tokenize_datasets(X, data.tokenizer)
    X_input_ids = data.trunc_n_pad(X_input_ids)
    X_masks = data.create_attention_masks(X_input_ids)
    #for i in range(len(X_input_ids)):
    #    print(len(X_input_ids[i]))
    #    print(len(X_masks[i]))
    #print(X_input_ids[0])
    #print(X_masks[0])
    inputs = torch.tensor(X_input_ids)
    masks = torch.tensor(X_masks)
    print(torch.cuda.is_available())
    labels = []
    device = torch.device('cuda:2')
    masked_input = TensorDataset(inputs, masks)
    dataloader = DataLoader(masked_input, batch_size=32)
    model.eval()
    model = model.to(device)
    mylogger.info('Model initializing')
    print("number of batches ",len(dataloader))
    
    start = datetime.now()
    for idx, batch in enumerate(dataloader):
        if idx%1000==0:
            mylogger.info(str(idx)+" "+str(datetime.now()-start))
            print(str(idx)+" "+str(datetime.now()-start))
        b_input, b_mask = batch
        b_input = b_input.to(device)
        b_mask = b_mask.to(device)
        #print(b_input.shape)
        #print(b_mask.shape)
        #print(model(b_input,b_mask).shape)
        labels.extend(model(b_input,b_mask).tolist())
    #print(labels)
    #model.eval()
    #print(type(model))
    #print(model(inputs,masks).shape)
    
    #sents = neg_data['data'].tolist()
    #sents = ["random sentence", "pretty flowers", "idiot", "fuck you cunt nigger"]
    #xs,masks = data.process(sents)
    #for sent in sents:
    #    x, mask = data.process(sent)
        #print(sent,' ',model(x, mask).item())
    #    labels.append(model(x, mask).item())
        
    neg_data["label"] = pd.Series(labels)
    #print(neg_data[["data","label","author"]])
    neg_data.to_csv(out_path)
    logging.shutdown()   

if __name__ == '__main__':
    main()

