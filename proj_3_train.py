#Lightning Imports
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

#Other Imports
import pandas as pd
import re
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import numpy as np
from transformers import BertForSequenceClassification
# from transformers import BerttokenizerFast
from transformers import AutoTokenizer
from pytorch_lightning.loggers import TensorBoardLogger
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
# from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune import JupyterNotebookReporter
# from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import TuneReportCallback,     TuneReportCheckpointCallback



class ProjData(pl.LightningDataModule):

    def __init__(self, data_dir: str = "~/tmp", batch_size: int = 32, max_len : int = 128, ratio : int = 2, adjacent=False, adjrat=None, adjtot=None):
        super().__init__()
        self.data_dir = data_dir
        # self.batch_size = batch_size
        self.max_len = max_len # Bert Max Len input
        self.rat = ratio
        if adjacent:
            self.name = "-".join(["b", str(adjacent), str(max_len), str(ratio), str(adjrat), str(adjtot)])
        else:
            self.name = "b-" + str(max_len) + "-" + str(ratio)
        self.tokenizer = None
        self.adjacent = adjacent
        self.adjrat = adjrat
        self.adjtot = adjtot

    def setup(self, stage=None):
        
        # *** tokenizer isn't actually a constant and the do_lower_case should be redundant if preprocessing was correct.
        # self.tokenizer = BerttokenizerFast.from_pretrained(MODEL_NAME, do_lower_case=False)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, do_lower_case=True)


        ###
        # Preprocessing Data
        ###
        print("Getting pos")
        if self.adjacent:
            pos_data = self.get_push_dataset(path="/bigtemp/rm5tx/nlp_project/adjacent_data/adjacent_2016-06_rate_" + str(self.adjrat) + "_tot_" + str(self.adjtot) + "_positive.csv", label=1)
            neg_data = self.get_push_dataset(path="/bigtemp/rm5tx/nlp_project/adjacent_data/adjacent_2016-06_rate_" + str(self.adjrat) + "_tot_" + str(self.adjtot) + "_negative.csv", label=0)
        else:
            pos_data = self.get_pos_dataset()
            neg_data = self.get_push_dataset()
        print("Getting neg")

        # *** TODO: Proper subset selection either in concat_datasets or in get_push_dataset
        print("Joining")
        dataset = self.concat_datasets(pos_data, neg_data)
        print(pos_data, neg_data, dataset)
        dataset.dropna(inplace=True) # Losing negative examples potentially...
        dataset["data"] = dataset["data"].map(self.preprocess)

        # 60% - train set, 20% - validation set, 20% - test set
        train, validate, test = np.split(dataset.sample(frac=1, random_state=42), 
                       [int(.6*len(dataset)), int(.8*len(dataset))])

        X_train, y_train = train["data"], train["label"]
        X_val, y_val = validate["data"], validate["label"]
        X_test, y_test = test["data"], test["label"]    


        

        ###
        # Tokenization
        ###
        # Convert texts into tokens. (These are not truncated or padded yet)
        print("Tokenizing")
        pre_train_input_ids = self.tokenize_datasets(X_train, self.tokenizer)
        pre_val_input_ids = self.tokenize_datasets(X_val, self.tokenizer)
        pre_test_input_ids = self.tokenize_datasets(X_test, self.tokenizer)
        
        # Truncate and Pad your tokens
        print("Padding")
        train_input_ids = self.trunc_n_pad(pre_train_input_ids)
        val_input_ids = self.trunc_n_pad(pre_val_input_ids)
        test_input_ids = self.trunc_n_pad(pre_test_input_ids)

        ###
        # Misc.
        ###
        # Create attention masks
        # print("Creating masks")
        # train_attention_masks = self.create_attention_masks(train_input_ids)
        # val_attention_masks = self.create_attention_masks(val_input_ids)
        # test_attention_masks = self.create_attention_masks(test_input_ids)

        # # Convert all of our data into torch tensors, the required datatype for our model
        train_inputs = torch.tensor(train_input_ids)
        validation_inputs = torch.tensor(val_input_ids)
        test_inputs = torch.tensor(test_input_ids)

        train_labels = torch.tensor(y_train.values.tolist())
        validation_labels = torch.tensor(y_val.values.tolist())
        test_labels = torch.tensor(y_test.values.tolist())

        # train_masks = torch.tensor(train_attention_masks)
        # validation_masks = torch.tensor(val_attention_masks)
        # test_masks = torch.tensor(test_attention_masks)

        train_masks = train_inputs.ne(0)
        validation_masks = validation_inputs.ne(0)
        test_masks = test_inputs.ne(0)

        # Create an iterator of our data with torch DataLoader. 
        self.train = TensorDataset(train_inputs, train_masks, train_labels)
        self.val = TensorDataset(validation_inputs, validation_masks, validation_labels)
        self.test = TensorDataset(test_inputs, test_masks, test_labels)
        print("pos counts:", torch.sum(train_labels), torch.sum(validation_labels), torch.sum(test_labels))


    def process(self, sent):
        s = self.preprocess(sent)
        s = self.tokenize_datasets([s], self.tokenizer)
        s = self.trunc_n_pad(s)
        # mask = self.create_attention_masks(s)
        s = torch.tensor(s)
        mask = s.ne(0)
        return  s, mask



    def preprocess(self, x):

        x = re.sub(r'(\n+)',r' ', x)
        # lower case in tokenizer
        x = " ".join(re.findall('[\w]+',x))
        return x
        
        # # RNV uses a special preprocess step --- mirror it for pushio.
        # # print("Preprocessing... 1. split new lines, 2. convert to lowercase, and 3. strip numbers and punct")
        # ### 1) remove newlines
        # data['data'] = data['data'].replace('\n', ' ', regex = True)

        # ## 2) convert to lowercase
        # data['data'] = data['data'].str.lower()

        # # ### 3) remove punct and numbers: https://stackoverflow.com/questions/47947438/preprocessing-string-data-in-pandas-dataframe
        # data["data"] = data.data.apply(lambda x : " ".join(re.findall('[\w]+',x)))
 

    # Import PushIO CSV    
    def get_push_dataset(self, path="/bigtemp/rm5tx/nlp_project/2016-05_all.csv", label=0):
        print("reading neg")
        if not self.adjacent:
            data = pd.read_csv(path, usecols=['body'], dtype="string")
            data.rename(columns={"body":"data"}, inplace=True)
        else:
            data = pd.read_csv(path, usecols=['data'], dtype="string")
        data["label"] = label

        return data

    # Reddit Norm Violations
    def get_pos_dataset(self, path="/bigtemp/rm5tx/nlp_project/reddit-norm-violations/data/macro-norm-violations/"):
        directory = os.path.abspath(path)

        pos_temp = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                with open(root+ "/" +file) as f:
                    pos_temp += f.readlines()
        data = pd.DataFrame(data=pos_temp, dtype = "string")
        data.rename(columns={0:"data"}, inplace=True)
        data["label"] = 1
        return data

    def concat_datasets(self, data_a, data_b):
        if len(data_a.index) > len(data_b.index):
            data_a, data_b = data_b, data_a
        # frames = [data_a, data_b[(self.rat * len(data_a.index)):]]
        trunced_b = data_b.sample(n=int(self.rat * len(data_a.index)), random_state=42)
        print("Using ", len(data_a.index), len(trunced_b.index), " samples.")
        dataset = pd.concat([data_a, trunced_b])
        print("Total = ", len(dataset.index))
        return dataset

    ###Pre-processing Code###
    def tokenize_datasets(self, X_dataset, tokenizer):
        input_ids = []
        for sent in X_dataset:
            tokenized_text = tokenizer.encode(
                                            sent,                      # Sentence to encode
                                            add_special_tokens = True, # Add '[CLS]' and '[SEP]' tokens
                                            max_length = self.max_len,      # Truncate senences
                                            truncation=True,
                                            )
            input_ids.append(tokenized_text)
        return input_ids


    def trunc_n_pad(self, input_id_list):

        # ret_list = []
        # for input_id in input_id_list:
        #     if len(input_id) > self.max_len:
        #         ret_list.append(input_id[:self.max_len])
        #     elif len(input_id) < self.max_len:
        #         temp_sublist = input_id + [0] * (self.max_len - len(input_id))
        #         ret_list.append(temp_sublist)
        #     else:
        #         ret_list.append(input_id)
        # return ret_list

        return [input_id[:self.max_len] + [0]*(self.max_len - len(input_id)) for input_id in input_id_list]

    #Create attention masks
    def create_attention_masks(self, input_ids):
        attention_masks = []
        for seq in input_ids:
            # Create a mask of 1s for each token followed by 0s for padding
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    def train_dataloader(self, batch_size=32):
        return DataLoader(self.train, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size=32):
        return DataLoader(self.val, batch_size=batch_size)

    def test_dataloader(self, batch_size=32):
        return DataLoader(self.test, batch_size=sbatch_size)

    def save(self, path="/bigtemp/rm5tx/nlp_project/data_cache/"):
        torch.save(self.tokenizer, open(path+self.name+"token.p", "wb"))
        torch.save(self.train, open(path+self.name+"train.pt", "wb"))
        torch.save(self.val, open(path+self.name+"val.pt", "wb"))
        torch.save(self.test, open(path+self.name+"test.pt", "wb"))

    def load(self, path="/bigtemp/rm5tx/nlp_project/data_cache/"):
        self.tokenizer = torch.load(open(path+self.name+"token.p", "rb"))
        self.train = torch.load(open(path+self.name+"train.pt", "rb"))
        self.val = torch.load(open(path+self.name+"val.pt", "rb"))
        self.test = torch.load(open(path+self.name+"test.pt", "rb"))    

class ProjModel(pl.LightningModule):
    def __init__(
        self,
        # model_name_or_path: str,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-9,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        # eval_splits: Optional[list] = None,
        **kwargs
        
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        # self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        # self.metric = ...

    def forward(self, x, mask):
        logits = self.model(x, attention_mask=mask).logits
        # return logits        
        pred = torch.argmax(logits, 1)
        return pred

    def training_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels = batch
        out = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        logits = out.logits
        loss = out.loss
        acc = self.accurate_nb(logits, b_labels)

        self.log('train_acc', acc)
        self.log('train_loss', loss)
        return loss

    # *** Old Monitor, NYI -- Do we want to use a scheduler at all?
    # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#automatic-optimization
    #     # for batch in validation_dataloader:
    #     #     with torch.no_grad():
    #     #         logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0] 

    #     #     tmp_eval_nb = accurate_nb(logits, b_labels)

    #     #     eval_accurate_nb += tmp_eval_nb
    #     #     nb_eval_examples += label_ids.shape[0]
    #     # eval_accuracy = eval_accurate_nb/nb_eval_examples
    #     # print("Validation Accuracy: {}".format(eval_accuracy))
    #     # scheduler.step(eval_accuracy)

    def validation_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels = batch
        loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels).loss
        logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask).logits
        
        acc = self.accurate_nb(logits, b_labels)

        self.log('valid_acc', acc)
        self.log('valid_loss', loss, on_step=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("avg_loss", avg_loss)
        self.log("avg_acc", avg_acc)   


    def test_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels = batch
        loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels).loss
        logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask).logits

        acc = self.accurate_nb(logits, b_labels)

        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        ###
        # Param Optim.
        ###
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': self.hparams.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'eval_acc'}
        return optimizer



    def accurate_nb(self, logits, labels):
        pred_flat = torch.argmax(logits, dim=1).flatten()
        labels_flat = labels.flatten()
        return torch.sum(pred_flat == labels_flat) / labels_flat.shape[0]



def main():
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32

    LEARNING_RATE = 1e-5
    MAX_EPOCHS = 20
    WEIGHT_DECAY = 0.0

    #DATA_PATH = "/bigtemp/rm5tx/nlp_project/data_cache/"
    DATA_PATH = os.path.expanduser("~/data_cache/")
    # DATA_NORM_PATH = os.path.expanduser("~/data_cache/")
    # DATA_ADJACENT_PATH = os.path.expanduser("~/data_adjacent_cache/")
    
    MAX_LEN = 128
    ADJACENT = True
    ADJRAT = 0.23
    ADJTOT = 2
    RATIO = 1


    data = ProjData(max_len=MAX_LEN, ratio=RATIO, adjacent=ADJACENT, adjrat=ADJRAT, adjtot=ADJTOT)
    if ADJACENT:
        # DATA_PATH = DATA_ADJACENT_PATH
        MODEL_NAME = 'nlp_proj_adjacent' + str(MAX_LEN)
    else:
        # DATA_PATH = DATA_NORM_PATH
        MODEL_NAME = 'nlp_proj_norm' + str(MAX_LEN) 

    try:
        data.load(DATA_PATH)
        print("Loaded Saved Data")
    except Exception as e: 
        print(e)
        data.setup()
        data.save(DATA_PATH)
    ### Comment out the try block and uncomment below while you're working on the data part or you'll just skip it and use old data.
    # data.setup() 
    # data.save(DATA_PATH)
    
    model = ProjModel(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    logger = TensorBoardLogger(os.path.expanduser("~/tb_logs"), name=MODEL_NAME)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', 
                                            dirpath=os.path.expanduser("~/saved_models"),
                                            save_last=True, 
                                            filename=MODEL_NAME + '-{epoch:02d}-{avg_acc:.2f}',)
    earlystopping = EarlyStopping(monitor='avg_acc', verbose=True, patience=0)

    trainer = pl.Trainer(logger=logger, 
                            accelerator='ddp', # jupyter can't use ddp, use dp instead
                            # effective batch size is batch_size * num_gpus * num_nodes
                            gpus=1, 
                            gradient_clip_val=1.0, 
                            max_epochs=MAX_EPOCHS,
                            fast_dev_run=False,
                            callbacks=[checkpoint_callback, earlystopping]
                        )
    trainer.fit(model, data.train_dataloader(batch_size=TRAIN_BATCH_SIZE), data.val_dataloader(batch_size=VAL_BATCH_SIZE))




if __name__ == '__main__':
   main()



 