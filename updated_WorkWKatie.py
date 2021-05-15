#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install keras matplotlib transformers numpy torch sklearn nltk pytorch-pretrained-bert pytorch-nlp


# In[2]:


#If there's a GPU available...
import torch

if torch.cuda.is_available():        
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# # Pre-processing Code

# In[4]:


# Import PushIO CSV
import pandas as pd

def get_pushio_dataset(path=""):
    if path:
        neg_data = pd.read_csv(path, usecols=['body'], dtype="string")
    else:
        neg_data = pd.read_csv("/bigtemp/rm5tx/nlp_project/2016-05_all.csv", usecols=['body'], dtype="string")
    
    # We want a unify col name for when we concat pos and neg data
    neg_data.rename(columns={"body":"data"}, inplace=True)
    neg_data["label"] = 0
    return neg_data


# In[5]:


# Reddit Norm Violations
import os
import re

def get_rnv_dataset(path=""):
    if path:
        directory = os.path.abspath(path)
    else:
        directory = os.path.abspath("/bigtemp/rm5tx/nlp_project/reddit-norm-violations/data/macro-norm-violations/")

    pos_temp = []
    for root,dirs,files in os.walk(directory):
        for file in files:
            with open(root+ "/" +file) as f:
                pos_temp += f.readlines()
    pos_data = pd.DataFrame(data=pos_temp, dtype = "string")
    pos_data.rename(columns={0:"data"}, inplace=True)
    pos_data["label"] = 1
    
    # RNV uses a special preprocess step
    print("Preprocessing... 1. split new lines, 2. convert to lowercase, and 3. strip numbers and punct")
    ### 1) remove newlines
    pos_data['data'] = pos_data['data'].replace('\n', ' ', regex = True)

    ## 2) convert to lowercase
    pos_data['data'] = pos_data['data'].str.lower()

    # ### 3) remove punct and numbers: https://stackoverflow.com/questions/47947438/preprocessing-string-data-in-pandas-dataframe
    pos_data["data"] = pos_data.data.apply(lambda x : " ".join(re.findall('[\w]+',x)))
    return pos_data


# In[6]:


def concat_datasets(data_a, data_b):
    frames = [data_a, data_b]
    dataset = pd.concat(frames)
    dataset.dropna(inplace=True)
    return dataset


# In[7]:


from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128    # Bert Max Len input
TOKENIZER = BertTokenizerFast.from_pretrained(MODEL_NAME, do_lower_case=True)

def tokenize_datasets(X_dataset, tokenizer, max_len=512):
    input_ids = []
    for sent in X_dataset:
        tokenized_text = tokenizer.encode(
                                        sent,                      # Sentence to encode
                                        add_special_tokens = True, # Add '[CLS]' and '[SEP]' tokens
                                        max_length = max_len,      # Truncate senences
                                        truncation=True,
                                        )
        input_ids.append(tokenized_text)
    return input_ids


# In[8]:


# Appears that CS Serv don[t have tf version 2.2]
# Thus, we cannot use the convenient pad_sequences from keras

def trunc_n_pad(input_id_list):
    ret_list = []
    for input_id in input_id_list:
        if len(input_id) > MAX_LEN:
            ret_list.append(input_id[:MAX_LEN])
        elif len(input_id) < MAX_LEN:
            temp_sublist = input_id + [0] * (MAX_LEN - len(input_id))
            ret_list.append(temp_sublist)
        else:
            ret_list.append(input_id)
    return ret_list


# In[9]:


# Create attention masks
def create_attention_masks(input_ids):
    attention_masks = []
    for seq in input_ids:
        # Create a mask of 1s for each token followed by 0s for padding
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


# In[10]:


import numpy as np

def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


# In[11]:


TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

LEARNING_RATE = 0.1
EPOCHS = 3
WEIGHT_DECAY = 0.2

SEED = 7


# In[ ]:


from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch import nn
from tqdm import trange 

def main():
    
    ###
    # Preprocessing Data
    ###
    neg_data = get_pushio_dataset()
    pos_data = get_rnv_dataset()
    dataset = concat_datasets(neg_data, pos_data)

    # 60% - train set, 20% - validation set, 20% - test set
    train, validate, test = np.split(dataset.sample(frac=1, random_state=42), 
                       [int(.6*len(dataset)), int(.8*len(dataset))])

    X_train, y_train = train["data"], train["label"]
    X_val, y_val = validate["data"], validate["label"]
    X_test, y_test = test["data"], test["label"]

    # NOTE: This is a small subset used for testing... likely will remove in final ver.
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_val = X_val[:1000]
    y_val = y_val[:1000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    ###
    # Tokenization
    ###
    # Convert texts into tokens. (These are not truncated or padded yet)
    pre_train_input_ids = tokenize_datasets(X_train, TOKENIZER, MAX_LEN)
    pre_val_input_ids = tokenize_datasets(X_val, TOKENIZER, MAX_LEN)
    pre_test_input_ids = tokenize_datasets(X_test, TOKENIZER, MAX_LEN)
    
    # Truncate and Pad your tokens
    train_input_ids = trunc_n_pad(pre_train_input_ids)
    val_input_ids = trunc_n_pad(pre_val_input_ids)
    test_input_ids = trunc_n_pad(pre_test_input_ids)

    ###
    # Misc.
    ###
    # Create attention masks
    train_attention_masks = create_attention_masks(train_input_ids)
    val_attention_masks = create_attention_masks(val_input_ids)
    test_attention_masks = create_attention_masks(test_input_ids)
    
    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_input_ids)
    validation_inputs = torch.tensor(val_input_ids)

    train_labels = torch.tensor(y_train.values.tolist())
    validation_labels = torch.tensor(y_val.values.tolist())

    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(val_attention_masks)

    test_inputs = torch.tensor(test_input_ids)
    test_labels = torch.tensor(y_test.values.tolist())

    test_masks = torch.tensor(test_attention_masks)
    
    # Create an iterator of our data with torch DataLoader. 
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
    
    # Create Dataloaders- a Python iterable over a dataset
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=VAL_BATCH_SIZE)

    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=TEST_BATCH_SIZE)
    
    
    ###
    # Model And Param Optim.
    ###
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]


    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    t_total = len(train_dataloader) * EPOCHS
    # Store our loss and accuracy for plotting

    best_val = -np.inf
    
    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(EPOCHS, desc="Epoch"): 
    # Training
        # Set our model to training mode (as opposed to evaluation mode)
        # Tracking variables
        tr_loss =  0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            loss_ce = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            if torch.cuda.device_count() > 1:
                loss_ce = loss_ce.mean()
            loss_ce.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss_ce.item()

            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train cross entropy loss: {}".format(tr_loss/nb_tr_steps))

        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_eval_examples = 0
        logits_list = []
        labels_list = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0] 
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)

            eval_accurate_nb += tmp_eval_nb
            nb_eval_examples += label_ids.shape[0]
        eval_accuracy = eval_accurate_nb/nb_eval_examples
        print("Validation Accuracy: {}".format(eval_accuracy))
        scheduler.step(eval_accuracy)


        if eval_accuracy > best_val:
            dirname = '{}/BERT-base-{}'.format(dataset, SEED)
            # Directory names at longest can be 255
            dirname = dirname[:255]
            output_dir = './model_save/{}'.format(dirname)
            os.makedirs(output_dir, exist_ok=True)
            print("Saving model to %s" % output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)   
            #tokenizer.save_pretrained(output_dir)

            best_val = eval_accuracy

    # ##### test model on test data
        # Put model in evaluation mode
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_test_examples = 0
        logits_list = []
        labels_list = []
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions 
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
            eval_accurate_nb += tmp_eval_nb
            nb_test_examples += label_ids.shape[0]

        print("Test Accuracy: {}".format(eval_accurate_nb/nb_test_examples))
main()


# In[ ]:


if __name__ == "__main__":
    main()

