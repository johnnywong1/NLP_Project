#%%
import torch
import pandas as pd
# import code
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

path = "/localtmp/rm5tx/nlp_project/data_cache/"
name = "a-128-2"

# tokenizer = torch.load(open(path+name+"token.p", "rb"))
train = torch.load(open(path+name+"train.pt", "rb"))
val = torch.load(open(path+name+"val.pt", "rb"))
test = torch.load(open(path+name+"test.pt", "rb"))   

# code.interact(local=locals())

# %%
# for b in train:
#     print(b[1])
#     break

# ut = [b[1] for b in train]
# %%
l1 = [l for batch in train for l in batch[1]]
l2 = [l for batch in test for l in batch[1]]
l3 = [l for batch in val for l in batch[1]]
# %%
from scipy import stats
print(stats.describe(l1))
print(stats.describe(l2))
print(stats.describe(l3))
# %%
