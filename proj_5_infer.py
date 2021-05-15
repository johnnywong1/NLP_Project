from proj_3_train import ProjModel, ProjData
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset,DataLoader
import multiprocessing as mp
from pandarallel import pandarallel
from time import ctime, time
#pip install pandarallel
import re

def get_dataset(path="/bigtemp/rm5tx/nlp_project/2016-06_all.csv"):
    print("reading neg")
    data = pd.read_csv(path, usecols=['body', 'author'], dtype={'body':str, 'author':str})
    data.rename(columns={"body":"data"}, inplace=True)
    data.dropna(subset=['data'],inplace=True)
    return data

def tprint(s):
    print(ctime(time()), ": ", s)

def tokenize(maxlen, tokenizer, x):
    x = re.sub(r'(\n+)',r' ', x)
    x = " ".join(re.findall('[\w]+',x))
    res = tokenizer.encode(
                    x,                      # Sentence to encode
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]' tokens
                    max_length = maxlen,      # Truncate senences
                    truncation=True,
                    )
    res = res[:maxlen] + [0]*(maxlen - len(res))
    return pd.array(res, dtype="Int64")



def main():
    
    out_path="/bigtemp/rm5tx/nlp_project/2016-06_all_predicted.csv"
    model = ProjModel.load_from_checkpoint(checkpoint_path=os.path.expanduser("~/saved_models/last.ckpt"))
    tprint("Model Loaded")
    DATA_PATH = os.path.expanduser("~/data_cache/")
    
    data = ProjData(max_len=128)
    data.load(DATA_PATH)
    tprint("Data Loaded")

    neg_data = get_dataset("/localtmp/rm5tx/2016-06_all.csv")

    # pool = mp.Pool(processes = (15)
    pandarallel.initialize(nb_workers=18)


    tprint("Starting Tokenize")
    neg_data['tokenized'] = neg_data['data'].parallel_map(lambda x: tokenize(data.max_len, data.tokenizer, x))
    tprint("Finished Tokenize")

    torch.save(neg_data, open( DATA_PATH+"df_to_be_inferred.pt", "wb"))
    neg_data.to_csv(DATA_PATH+"df_to_be_inferred.csv")

    inputs = torch.tensor(neg_data['tokenized'])
    masks = inputs.ne(0)

    neg_data.to_csv(DATA_PATH+"df_to_be_inferred.csv")
    torch.save(inputs, open( DATA_PATH+"inputs_to_be_inferred.pt", "wb"))

    

    tprint("Saved Inputs")

    labels = []
    masked_input = TensorDataset(inputs, masks)
    dataloader = DataLoader(masked_input, batch_size=1000)
    model.eval()
    for batch in dataloader:
        b_input, b_mask = batch
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
                      

if __name__ == '__main__':
    main()

